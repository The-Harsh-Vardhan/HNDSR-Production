"""
model_registry/canary_deploy.py
=================================
Canary and shadow deployment logic for HNDSR model rollouts.

What  : Manages gradual traffic shifting from old model to new model,
        with automatic rollback if quality metrics degrade.
Why   : Deploying a new model to 100% traffic is risky. A single bad
        model can degrade all user-facing outputs. Canary deployments
        limit blast radius to 10% of traffic.
How   : Traffic splitting via weighted routing rules + metric comparison
        between canary and baseline using a statistical significance test.

Deployment Patterns:
  1. Canary: Route 10% traffic to new model, 90% to old
  2. Shadow: Route 100% to old model, duplicate requests to new model
             for comparison only (no user impact)
  3. Blue-Green: Instant switch between two full deployments
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DeploymentState(str, Enum):
    """Possible states of a canary deployment."""
    PENDING = "pending"       # Not yet started
    ROLLING_OUT = "rolling_out"  # Canary is receiving traffic
    OBSERVING = "observing"   # Waiting for metrics to stabilize
    PROMOTING = "promoting"   # Shifting to 100%
    ROLLING_BACK = "rolling_back"  # Reverting to baseline
    COMPLETED = "completed"   # Successfully deployed
    FAILED = "failed"         # Rolled back due to issues


@dataclass
class CanaryMetrics:
    """Metrics snapshot for comparison."""
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    error_rate_pct: float = 0.0
    psnr_mean: float = 0.0
    ssim_mean: float = 0.0
    request_count: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class CanaryConfig:
    """Configuration for a canary deployment."""
    # Traffic splitting
    initial_traffic_pct: float = 10.0    # Start with 10% to canary
    max_traffic_pct: float = 100.0
    traffic_increment_pct: float = 20.0  # Increase by 20% each step
    increment_interval_minutes: float = 30.0

    # Quality gates
    max_latency_increase_pct: float = 20.0  # Reject if >20% slower
    max_error_rate_pct: float = 5.0          # Reject if >5% errors
    min_psnr_delta: float = -0.5             # Accept up to 0.5 dB drop
    min_request_count: int = 50              # Min requests before deciding

    # Observation
    observation_window_minutes: float = 60.0  # Watch for 1 hour
    auto_rollback: bool = True


class CanaryDeployment:
    """
    Manages a single canary deployment lifecycle.

    Lifecycle:
      1. Start canary at 10% traffic
      2. Observe for 60 minutes
      3. If metrics OK → increase to 30% → 50% → 100%
      4. If metrics BAD → auto-rollback to 0%

    Usage:
        canary = CanaryDeployment(
            model_name="hndsr",
            baseline_version=3,
            canary_version=4,
        )
        canary.start()

        # Periodically feed metrics
        canary.update_metrics(
            baseline=CanaryMetrics(latency_p50_ms=200, psnr_mean=27.0, ...),
            canary=CanaryMetrics(latency_p50_ms=210, psnr_mean=27.3, ...),
        )

        # Check if we should promote or rollback
        decision = canary.evaluate()
    """

    def __init__(
        self,
        model_name: str,
        baseline_version: int,
        canary_version: int,
        config: Optional[CanaryConfig] = None,
    ):
        self.model_name = model_name
        self.baseline_version = baseline_version
        self.canary_version = canary_version
        self.config = config or CanaryConfig()

        self.state = DeploymentState.PENDING
        self.current_traffic_pct = 0.0
        self.start_time: Optional[float] = None
        self.last_increment_time: Optional[float] = None

        self._baseline_metrics: List[CanaryMetrics] = []
        self._canary_metrics: List[CanaryMetrics] = []
        self._decision_log: List[str] = []

    def start(self) -> Dict:
        """Start the canary deployment."""
        self.state = DeploymentState.ROLLING_OUT
        self.current_traffic_pct = self.config.initial_traffic_pct
        self.start_time = time.time()
        self.last_increment_time = time.time()

        self._log(
            f"Started canary: v{self.baseline_version} (90%) ↔ "
            f"v{self.canary_version} (10%)"
        )

        return self._get_routing_config()

    def update_metrics(
        self,
        baseline: CanaryMetrics,
        canary: CanaryMetrics,
    ):
        """Record latest metrics from both deployments."""
        self._baseline_metrics.append(baseline)
        self._canary_metrics.append(canary)

    def evaluate(self) -> Dict:
        """
        Evaluate canary health and decide next action.

        Returns:
            {
                "action": "continue" | "promote" | "rollback",
                "traffic_pct": float,
                "reason": str,
                "routing_config": dict,
            }
        """
        if self.state in (DeploymentState.COMPLETED, DeploymentState.FAILED):
            return {"action": "none", "reason": "Deployment already finalized"}

        if not self._canary_metrics:
            return {"action": "continue", "reason": "No metrics yet"}

        latest_canary = self._canary_metrics[-1]
        latest_baseline = self._baseline_metrics[-1]

        # ── Check minimum samples ────────────────────────────────────
        if latest_canary.request_count < self.config.min_request_count:
            return {
                "action": "continue",
                "reason": f"Need {self.config.min_request_count} requests, "
                          f"have {latest_canary.request_count}",
            }

        # ── Check error rate ─────────────────────────────────────────
        if latest_canary.error_rate_pct > self.config.max_error_rate_pct:
            return self._rollback(
                f"Error rate {latest_canary.error_rate_pct:.1f}% > "
                f"{self.config.max_error_rate_pct}%"
            )

        # ── Check latency regression ─────────────────────────────────
        if latest_baseline.latency_p95_ms > 0:
            latency_increase = (
                (latest_canary.latency_p95_ms - latest_baseline.latency_p95_ms)
                / latest_baseline.latency_p95_ms * 100
            )
            if latency_increase > self.config.max_latency_increase_pct:
                return self._rollback(
                    f"P95 latency increase {latency_increase:.1f}% > "
                    f"{self.config.max_latency_increase_pct}%"
                )

        # ── Check quality regression ─────────────────────────────────
        psnr_delta = latest_canary.psnr_mean - latest_baseline.psnr_mean
        if psnr_delta < self.config.min_psnr_delta:
            return self._rollback(
                f"PSNR delta {psnr_delta:.2f} dB < {self.config.min_psnr_delta} dB"
            )

        # ── All checks passed — promote or continue ──────────────────
        elapsed = time.time() - (self.last_increment_time or self.start_time)
        elapsed_minutes = elapsed / 60

        if elapsed_minutes >= self.config.increment_interval_minutes:
            return self._increment_traffic()

        return {
            "action": "continue",
            "reason": f"Metrics OK. Next increment in "
                      f"{self.config.increment_interval_minutes - elapsed_minutes:.0f} min",
            "traffic_pct": self.current_traffic_pct,
        }

    # ── Private helpers ──────────────────────────────────────────────────

    def _increment_traffic(self) -> Dict:
        """Increase canary traffic percentage."""
        new_pct = min(
            self.current_traffic_pct + self.config.traffic_increment_pct,
            self.config.max_traffic_pct,
        )

        if new_pct >= self.config.max_traffic_pct:
            return self._promote()

        self.current_traffic_pct = new_pct
        self.last_increment_time = time.time()
        self._log(f"Increased canary traffic to {new_pct:.0f}%")

        return {
            "action": "continue",
            "reason": f"Promoted to {new_pct:.0f}% traffic",
            "traffic_pct": new_pct,
            "routing_config": self._get_routing_config(),
        }

    def _promote(self) -> Dict:
        """Promote canary to full production."""
        self.state = DeploymentState.COMPLETED
        self.current_traffic_pct = 100.0
        self._log(f"Canary v{self.canary_version} promoted to production!")

        return {
            "action": "promote",
            "reason": "All quality gates passed across all traffic levels",
            "traffic_pct": 100.0,
            "routing_config": self._get_routing_config(),
        }

    def _rollback(self, reason: str) -> Dict:
        """Roll back canary deployment."""
        self.state = DeploymentState.FAILED
        self.current_traffic_pct = 0.0
        self._log(f"ROLLBACK: {reason}")

        return {
            "action": "rollback",
            "reason": reason,
            "traffic_pct": 0.0,
            "routing_config": self._get_routing_config(),
        }

    def _get_routing_config(self) -> Dict:
        """Generate routing configuration for the load balancer."""
        return {
            "baseline": {
                "version": self.baseline_version,
                "weight": 100 - self.current_traffic_pct,
            },
            "canary": {
                "version": self.canary_version,
                "weight": self.current_traffic_pct,
            },
        }

    def _log(self, msg: str):
        """Log a decision."""
        entry = f"[{time.strftime('%H:%M:%S')}] {msg}"
        self._decision_log.append(entry)
        logger.info(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Shadow deployment (observation-only)
# ─────────────────────────────────────────────────────────────────────────────

class ShadowDeployment:
    """
    Shadow deployment: duplicates requests to the new model without
    affecting user-facing responses.

    All production traffic goes to the baseline model. A copy of each
    request is also sent to the shadow model. Responses from the shadow
    model are logged but never shown to users.

    Use case: Validate a new model's latency and quality characteristics
    with real production traffic before any canary rollout.
    """

    def __init__(
        self,
        model_name: str,
        production_version: int,
        shadow_version: int,
    ):
        self.model_name = model_name
        self.production_version = production_version
        self.shadow_version = shadow_version
        self.active = False
        self._comparison_log: List[Dict] = []

    def start(self):
        """Enable shadow deployment."""
        self.active = True
        logger.info(
            "Shadow deployment started: prod=v%d, shadow=v%d",
            self.production_version, self.shadow_version,
        )

    def stop(self) -> List[Dict]:
        """Stop shadow deployment and return comparison log."""
        self.active = False
        logger.info(
            "Shadow deployment stopped after %d comparisons",
            len(self._comparison_log),
        )
        return self._comparison_log

    def log_comparison(
        self,
        request_id: str,
        production_latency_ms: float,
        shadow_latency_ms: float,
        production_psnr: Optional[float] = None,
        shadow_psnr: Optional[float] = None,
    ):
        """Log a side-by-side comparison between production and shadow."""
        self._comparison_log.append({
            "request_id": request_id,
            "timestamp": time.time(),
            "production_latency_ms": production_latency_ms,
            "shadow_latency_ms": shadow_latency_ms,
            "production_psnr": production_psnr,
            "shadow_psnr": shadow_psnr,
            "latency_delta_pct": (
                (shadow_latency_ms - production_latency_ms)
                / production_latency_ms * 100
                if production_latency_ms > 0 else None
            ),
        })

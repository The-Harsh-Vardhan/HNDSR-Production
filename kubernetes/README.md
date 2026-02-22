# HNDSR Kubernetes Orchestration

This directory contains manifests for deploying the HNDSR system to a Kubernetes cluster.

## Manifests
- `deployment.yaml`: GPU-scheduled deployment configurations.
- `service.yaml`: Load balancer and internal service definitions.
- `hpa.yaml`: Horizontal Pod Autoscaler for inference workers.
- `pdb.yaml`: Pod Disruption Budgets for high availability.

"""Quick test script to diagnose /infer endpoint errors."""
import base64
import json
import urllib.request
import sys

# Read the sample image
img_path = r"tests\Sample Images\HG_Satellite_LoRes_Pic1_TerraColor.avif"
try:
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    print(f"Image loaded: {len(img_b64)} chars base64")
except FileNotFoundError:
    print(f"ERROR: Image not found at {img_path}")
    sys.exit(1)

# Send inference request
payload = {
    "image": img_b64,
    "scale_factor": 4,
    "ddim_steps": 20,
    "return_metadata": True,
}

data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(
    "http://localhost:8000/infer",
    data=data,
    headers={"Content-Type": "application/json"},
)

try:
    resp = urllib.request.urlopen(req, timeout=300)
    result = json.loads(resp.read())
    meta = result.get("metadata", {})
    print(f"SUCCESS!")
    print(f"  Output: {result.get('width')}x{result.get('height')}")
    print(f"  Scale: {result.get('scale_factor')}")
    print(f"  Latency: {meta.get('latency_ms')} ms")
    print(f"  Device: {meta.get('device')}")
    print(f"  Model: {meta.get('model')}")
    print(f"  FP16: {meta.get('fp16')}")
    print(f"  Image data length: {len(result.get('image', ''))} chars")
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8", errors="replace")
    print(f"HTTP ERROR {e.code}: {e.reason}")
    print(f"Response body: {body}")
except Exception as e:
    print(f"ERROR: {e}")

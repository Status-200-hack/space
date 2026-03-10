"""
Seed script: seeds 12 satellites + 200 debris into the live sim registry via
POST /api/v1/registry/satellites and POST /api/v1/registry/debris.
Then verifies the /snapshot endpoint returns them all.
"""
import math, random, json
import urllib.request, urllib.error

BASE = "http://localhost:8000/api/v1"
RE   = 6371.0  # Earth radius km

# ─── Helpers ─────────────────────────────────────────────────────────────────

def post_json(url, data):
    body = json.dumps(data).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()

def get_json(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())

def eci(lat_deg, lon_deg, alt_km):
    """Approximate ECI position + circular orbital velocity from geodetic."""
    r    = RE + alt_km
    lat  = math.radians(lat_deg)
    lon  = math.radians(lon_deg)
    x    = r * math.cos(lat) * math.cos(lon)
    y    = r * math.cos(lat) * math.sin(lon)
    z    = r * math.sin(lat)
    v    = math.sqrt(398600.4418 / r)   # km/s circular speed
    vx   = -v * math.sin(lon)
    vy   =  v * math.cos(lon)
    vz   = 0.0
    return x, y, z, vx, vy, vz


# ─── 12 Satellites ───────────────────────────────────────────────────────────

SAT_DATA = [
    ("STAR-1",  28.5, -80.0,  550, 65.0, "active"),
    ("STAR-2",  45.0,  20.0,  560, 58.3, "active"),
    ("STAR-3", -30.0, 100.0,  540, 72.1, "active"),
    ("STAR-4",  60.0, -45.0,  575, 12.4, "active"),
    ("STAR-5",   0.0, 150.0,  530, 80.0, "active"),
    ("STAR-6",  15.0,-120.0,  565,  4.9, "active"),
    ("STAR-7", -50.0, -60.0,  520, 55.0, "active"),
    ("STAR-8",  75.0,  90.0,  590, 61.7, "active"),
    ("STAR-9",  30.0,  60.0,  545, 18.2, "active"),
    ("STAR-10",-15.0,-150.0,  535, 90.0, "active"),
    ("STAR-11", 50.0, 170.0,  580, 47.3, "active"),
    ("STAR-12",-70.0,  30.0,  555, 33.6, "active"),
]

print("Seeding satellites → POST /api/v1/registry/satellites")
sat_ok = 0
for name, lat, lon, alt, fuel, status in SAT_DATA:
    x, y, z, vx, vy, vz = eci(lat, lon, alt)
    payload = {
        "name": name,
        "position": {"x": x,   "y": y,   "z": z,   "vx": 0.0, "vy": 0.0, "vz": 0.0},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0, "vx": vx,  "vy": vy,  "vz": vz},
        "fuel_kg": fuel,
        "status":  status,
    }
    code, resp = post_json(f"{BASE}/registry/satellites", payload)
    if code == 201:
        sat_ok += 1
        print(f"  ✓ {name} ({status}, {fuel} kg fuel)")
    else:
        print(f"  ✗ {name} → HTTP {code}: {str(resp)[:80]}")

# ─── 200 Debris ──────────────────────────────────────────────────────────────

print(f"\nSeeding 200 debris → POST /api/v1/registry/debris")
random.seed(42)
deb_ok = 0
for i in range(200):
    lat = random.uniform(-75, 75)
    lon = random.uniform(-180, 180)
    alt = random.uniform(380, 720)
    x, y, z, vx, vy, vz = eci(lat, lon, alt)
    payload = {
        "designation": f"DEB-{i:04d}",
        "position": {"x": x,   "y": y,   "z": z,   "vx": 0.0, "vy": 0.0, "vz": 0.0},
        "velocity": {"x": 0.0, "y": 0.0, "z": 0.0, "vx": vx,  "vy": vy,  "vz": vz},
        "radar_cross_section_m2": round(random.uniform(0.01, 1.5), 3),
    }
    code, resp = post_json(f"{BASE}/registry/debris", payload)
    if code == 201:
        deb_ok += 1
    else:
        print(f"  ✗ DEB-{i:04d} → HTTP {code}: {str(resp)[:60]}")

print(f"  ✓ {deb_ok}/200 debris objects created")

# ─── Verify snapshot ─────────────────────────────────────────────────────────

print("\nVerifying GET /api/v1/visualization/snapshot ...")
snap = get_json(f"{BASE}/visualization/snapshot")
print(f"  timestamp       : {snap.get('timestamp')}")
print(f"  satellite_count : {snap.get('satellite_count')}")
print(f"  debris_count    : {snap.get('debris_count')}")
print(f"  snapshot_time   : {snap.get('meta', {}).get('snapshot_time_ms')} ms")
print()
print("Done! Open http://localhost:3001 – the dashboard should now show live data.")

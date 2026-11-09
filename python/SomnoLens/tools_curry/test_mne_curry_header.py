"""
Run after upgrading the conda env (MNE 1.12 + curryreader) to decide whether
curry_header.py is needed.

Checks two criteria:
  1. Speed  : read_raw_curry(preload=False) must be instantaneous (< 2 s)
              — if it loads the 7.6 GB signal it will take minutes and fail this test.
  2. Completeness : all header fields needed by the Curry tools must be accessible
              from raw.info / MNE internals without signal loading.

Run:
    & "$env:LOCALAPPDATA\miniforge3\envs\somnolens\python.exe" tools_curry/test_mne_curry_header.py
"""

import time
import mne

CDT = r"tools/test_data_curry/y_S005/y_S005.cdt"

print("=" * 60)
print("MNE version:", mne.__version__)
print("=" * 60)

# --- Criterion 1: speed ---
t0 = time.time()
raw = mne.io.read_raw_curry(CDT, preload=False, verbose="ERROR")
elapsed = time.time() - t0
speed_ok = elapsed < 2.0
print(f"\n[SPEED] read_raw_curry(preload=False) took {elapsed:.2f} s  →  {'OK' if speed_ok else 'SLOW — full data load suspected'}")

# --- Criterion 2: completeness ---
print("\n[COMPLETENESS] Checking fields needed by Curry tools:")

checks = {}

# sfreq
checks["sfreq"] = raw.info["sfreq"]
print(f"  sfreq            : {raw.info['sfreq']} Hz")

# meas_date
checks["meas_date"] = raw.info["meas_date"]
print(f"  meas_date        : {raw.info['meas_date']}")

# n_times / duration
print(f"  n_times          : {raw.n_times}  ({raw.n_times / raw.info['sfreq'] / 3600:.2f} h)")

# channel names + types
ch_types = raw.get_channel_types()
type_counts = {t: ch_types.count(t) for t in sorted(set(ch_types))}
checks["ch_names"] = raw.ch_names
print(f"  ch_names ({len(raw.ch_names)} ch) : {raw.ch_names[:5]} … {raw.ch_names[-3:]}")
print(f"  ch_types         : {type_counts}")

# channel units (DataUnit = uV)
try:
    units = [raw.info["chs"][i].get("unit", "?") for i in range(min(3, len(raw.ch_names)))]
    print(f"  ch units (first 3): {units}")
    checks["units"] = units
except Exception as e:
    print(f"  ch units         : ERROR — {e}")
    checks["units"] = None

# --- Impedances --- the critical unknown
print("\n  [IMPEDANCES] Searching raw._raw_extras and raw.info for impedance data …")
impedances_found = False

# Try _raw_extras (MNE internal)
extras = getattr(raw, "_raw_extras", None)
if extras:
    print(f"    _raw_extras keys: {list(extras[0].keys()) if isinstance(extras, list) else list(extras.keys())}")
    for key in (extras[0] if isinstance(extras, list) else extras):
        if "imped" in key.lower() or "imp" in key.lower():
            print(f"    → Found impedance key: '{key}'")
            impedances_found = True
else:
    print("    _raw_extras: not present")

# Try raw.info['hpi_meas'] or other standard fields
for field in ["hpi_meas", "hpi_results", "hpi_subsystem", "device_info", "helium_info"]:
    val = raw.info.get(field)
    if val:
        print(f"    raw.info['{field}'] = {val!r:.120}")

if not impedances_found:
    print("    → Impedances NOT found in MNE internals (curry_header.py needed for this field)")

# --- Summary ---
print("\n" + "=" * 60)
print("SUMMARY")
print(f"  Speed OK      : {speed_ok}  ({elapsed:.2f} s)")
print(f"  Impedances    : {'found in MNE' if impedances_found else 'NOT in MNE — need curry_header.py'}")
print()
if speed_ok and impedances_found:
    print("→ curry_header.py NOT needed: MNE covers everything.")
elif speed_ok and not impedances_found:
    print("→ curry_header.py needed for IMPEDANCES only (MNE fast enough for other fields).")
else:
    print("→ curry_header.py needed for EVERYTHING (MNE too slow for header-only scan).")
print("=" * 60)

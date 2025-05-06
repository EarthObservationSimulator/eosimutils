import numpy as np

from eosimutils.frames import ReferenceFrame
from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.trajectory import StateSeries
from eosimutils.attitude import AttitudeSeries
from eosimutils.frame_registry import FrameRegistry

# 1) Build a circular orbit in ICRF_EC
μ = 398600.4418           # Earth's GM, km³/s²
r0 = 7000.0               # orbital radius, km
omega = np.sqrt(μ / r0**3)    # orbital rate, rad/s

# sample times: 0 to 3600 s in 10 steps
et_array = np.linspace(0.0, 3600.0, 10)
times = AbsoluteDateArray(et_array)

# positions & velocities in ICRF_EC
pos_icrf = np.vstack([
    r0 * np.cos(omega * et_array),
    r0 * np.sin(omega * et_array),
    np.zeros_like(et_array)
]).T

vel_icrf = np.vstack([
    -r0 * omega * np.sin(omega * et_array),
     r0 * omega * np.cos(omega * et_array),
     np.zeros_like(et_array)
]).T

state_icrf = StateSeries(
    time=times,
    data=[pos_icrf, vel_icrf],
    frame=ReferenceFrame.ICRF_EC
)

# 2) Compute the LVLH attitude series
att_lvlh = AttitudeSeries.get_lvlh(state_icrf)

# 3) Register the LVLH frame
lvlh_frame = ReferenceFrame.add("LVLH")
registry   = FrameRegistry()
registry.add_transform(
    ReferenceFrame.ICRF_EC,
    lvlh_frame,
    att_lvlh
)

# 4) Batch‐transform into LVLH
rot_array, w_array = registry.get_transform(
    ReferenceFrame.ICRF_EC,
    lvlh_frame,
    times
)
pos_lvlh = rot_array.apply(pos_icrf)
vel_lvlh = rot_array.apply(vel_icrf) + np.cross(w_array, pos_lvlh)

state_lvlh = StateSeries(
    time=times,
    data=[pos_lvlh, vel_lvlh],
    frame=lvlh_frame
)

# In LVLH, position should be ~[r0,0,0], velocity ~[0,0,0]
for t, p_l in zip(et_array, pos_lvlh):
    print(f"t={t:7.1f} s   LVLH pos ≈ {p_l.round(3)}")

for t, v_l in zip(et_array, vel_lvlh):
    print(f"t={t:7.1f} s   LVLH vel ≈ {v_l.round(3)}")
    
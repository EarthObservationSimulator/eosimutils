import numpy as np

from eosimutils.frames import ReferenceFrame
from eosimutils.time import AbsoluteDate, AbsoluteDateArray
from eosimutils.trajectory import StateSeries
from eosimutils.orientation import OrientationSeries
from eosimutils.frame_registry import FrameRegistry

# Build a circular orbit in ICRF_EC
μ = 398600.4418           # Earth's GM, km³/s²
r0 = 7000.0               # orbital radius, km
omega = np.sqrt(μ / r0**3)    # orbital rate, rad/s

# sample times: 0 to 3600 s in 10 steps
et_array = np.linspace(0.0, 3600.0, 10)
times = AbsoluteDateArray(et_array)

# positions & velocities in ICRF_EC
# this is a circular, prograde equatorial orbit
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
    frame=ReferenceFrame.get("ICRF_EC")
)

# Add LVLH frame
lvlh_frame = ReferenceFrame.add("LVLH")
att_lvlh = OrientationSeries.get_lvlh(state_icrf,lvlh_frame)

registry   = FrameRegistry()
registry.add_transform(att_lvlh)

# Batch‐transform into LVLH
rot_array, w_array = registry.get_transform(
    ReferenceFrame.get("ICRF_EC"),
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

# In LVLH, position should be ~[0,0,-r0], velocity ~[0,0,0]
for t, p_l in zip(et_array, pos_lvlh):
    print(f"t={t:7.1f} s   LVLH pos ≈ {p_l.round(3)}")

for t, v_l in zip(et_array, vel_lvlh):
    print(f"t={t:7.1f} s   LVLH vel ≈ {v_l.round(3)}")
    
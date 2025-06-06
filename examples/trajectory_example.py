import numpy as np
import matplotlib.pyplot as plt

from eosimutils.trajectory import StateSeries
from eosimutils.base import ReferenceFrame
from eosimutils.plotting import plot_timeseries
from eosimutils.time import AbsoluteDateArray

# Generate two sample trajectories representing circular orbits.

# Define orbit parameters.
R = 7000             # km, orbital radius for Orbit 1.
period = 5400.0/86400.0        # days, orbital period (~1.5 hours).
omega = 2 * np.pi / period  # angular velocity in rad/s.
jd_of_j2000 = 2451545.0  # Julian Date of J2000 epoch.

# Create an array of time samples.
N = 200
time_points = np.linspace(jd_of_j2000, jd_of_j2000 + period, N)
abs_dates = AbsoluteDateArray.from_dict({
    "time_format": "Julian_Date",
    "jd": time_points.tolist(),
    "time_scale": "UTC"
})

# Orbit 1: Circular orbit in the xy-plane.
x1 = R * np.cos(omega * time_points)
y1 = R * np.sin(omega * time_points)
z1 = np.full_like(time_points, 0)
vx1 = -R * omega * np.sin(omega * time_points)
vy1 = R * omega * np.cos(omega * time_points)
vz1 = np.full_like(time_points, 0)

position1 = np.column_stack((x1, y1, z1))
velocity1 = np.column_stack((vx1, vy1, vz1))

# Introduce missing data (NaN values) in Orbit 1 for a specific time range.
missing_start = int(0.3 * N)  # Start of missing data (30% of the way through).
missing_end = int(0.5 * N)    # End of missing data (50% of the way through).
position1[missing_start:missing_end, :] = np.nan
velocity1[missing_start:missing_end, :] = np.nan

traj1 = StateSeries.from_dict({
    "time": abs_dates.to_dict(),
    "data": [position1.tolist(), velocity1.tolist()],
    "frame": ReferenceFrame.get("ICRF_EC").to_string(),
    "headers": [["pos_x", "pos_y", "pos_z"], ["vel_x", "vel_y", "vel_z"]]
})
traj1_itrf = traj1.to_frame(ReferenceFrame.get("ITRF"))  # Convert to ITRF frame.

# Orbit 2: Nearly identical orbit with a small increase in radius.
dR = 1.0             # km difference.
R2 = R + dR
x2 = R2 * np.cos(omega * time_points)
y2 = R2 * np.sin(omega * time_points)
z2 = np.full_like(time_points, 0)
vx2 = -R2 * omega * np.sin(omega * time_points)
vy2 = R2 * omega * np.cos(omega * time_points)
vz2 = np.full_like(time_points, 0)

position2 = np.column_stack((x2, y2, z2))
velocity2 = np.column_stack((vx2, vy2, vz2))
traj2 = StateSeries.from_dict({
    "time": abs_dates.to_dict(),
    "data": [position2.tolist(), velocity2.tolist()],
    "frame": ReferenceFrame.get("ICRF_EC").to_string(),
    "headers": [["pos_x", "pos_y", "pos_z"], ["vel_x", "vel_y", "vel_z"]]
})

# Compute the difference between the two trajectories
traj_diff = traj1 - traj2

# Plot the trajectories.
# For clarity, we only plot the position columns (columns 0,1,2).

# Orbit 1 positions.
fig1, ax1, _ = plot_timeseries(traj1, cols=[0, 1, 2],
                               title="Orbit 1 (Positions)",
                               ylabel="Distance (km)")

# Orbit 1 positions in ITRF frame.
fig1_itrf, ax1_itrf, _ = plot_timeseries(traj1_itrf, cols=[0, 1, 2],
                                          title="Orbit 1 (ITRF Positions)",
                                          ylabel="Distance (km)")

# Orbit 2 positions.
fig2, ax2, _ = plot_timeseries(traj2, cols=[0, 1, 2],
                               title="Orbit 2 (Positions)",
                               ylabel="Distance (km)")
# Difference between orbits (positions only).
fig3, ax3, _ = plot_timeseries(traj_diff, cols=[0, 1, 2],
                               title="Difference between Orbits (Positions)",
                               ylabel="Difference (km)")

plt.show()
# eosimutils
Common utilities for packages and scripts within the EarthObservationSimulator organization

**Currently under active development.**

## Overview

**eosimutils** is a Python package providing a core set of utilities for space mission simulation. It is designed to support reference frame management, state and time handling, kinematic transformations, field-of-view modeling, and more. The package is modular, extensible, and leverages third-party libraries such as Spiceypy, NumPy, SciPy, Astropy, and Skyfield for scientific computation and astronomical calculations.

## Installation

Requires: Unix-like operating system, `python 3.13`, `pip`

Create a conda environment:
```
conda create -n eosim-revised python=3.13
conda activate eosim-revised
conda install sphinx
pip install sphinx-rtd-theme
pip install pylint
pip install black
pip install coverage
pip install skyfield
pip install astropy
pip install scipy

make install
```

### Main Features

- **Time Handling**  
  Handling of time using `AbsoluteDate` and `AbsoluteDateArray`, supporting conversions between (Spice) ephemeris time, Julian date, Gregorian date, and integration with Astropy and Skyfield.
  
```
eosimutils.time
    - TimeFormat
    - TimeScale
    - AbsoluteDate
    - AbsoluteDateArray
```

- **State Vector Representation**  
  Classes for representing and manipulating Cartesian positions, velocities, and full state vectors, including geodetic positions.

```
eosimutils.state
    - Cartesian3DPosition
    - Cartesian3DVelocity
    - GeographicPosition
    - CartesianState
    - Cartesian3DPositionArray
```

- **Trajectory and Timeseries**  
  Efficient storage and manipulation of time series data (e.g., position, state as a function of time), with support for interpolation, resampling, and missing data handling.

```
eosimutils.timeseries
    - Timeseries
    - _group_contiguous (function)

eosimutils.trajectory
    - StateSeries
    - PositionSeries
    - convert_frame (function)
    - convert_frame_position (function)
```

- **Reference Frame Management**  
  Define, register, and manage reference frames using the `ReferenceFrame` class and the `FrameRegistry`.

```
eosimutils.frame_registry
    - FrameRegistry

eosimutils.base
    - ReferenceFrame
```

- **Orientation (Attitude)**  
  Flexible orientation representations for modeling frame-to-frame transformations with consideration of rotating frames.
  Transforming positions and states between reference frames, including support for SPICE-based transformations.

```
eosimutils.orientation
    - Orientation
        - ConstantOrientation
        - SpiceOrientation
        - OrientationSeries
```

- **Field-of-View Modeling**  
  Classes for circular, rectangular, and polygonal field-of-view geometries, with a factory for easy instantiation.

```
eosimutils.fieldofview
    - FieldOfViewType
    - FieldOfViewFactory
    - CircularFieldOfView
    - RectangularFieldOfView
    - PolygonFieldOfView
```

- **SPICE Kernel Management**  
  Utilities for downloading and loading SPICE kernels required for time and frame conversions.

```
eosimutils.spicekernels
    - download_latest_kernels (function)
    - load_spice_kernels (function)
```

- **Plotting and Visualization**  
  Functions for plotting timeseries and trajectory data for analysis and presentation.

```
eosimutils.plotting
    - plot_timeseries (function)
```

- **Third-Party Utilities**  
  Wrappers and helpers for using Astropy and other libraries for validation and cross-checking.

```
eosimutils.thirdpartyutils
    - astropy_transform (function)
```



=============================
```
eosimutils.base
    - EnumBase
    - RotationsType
    - ReferenceFrame
```

```
eosimutils.kinematic
- transform_position (function)
- transform_state (function)
```



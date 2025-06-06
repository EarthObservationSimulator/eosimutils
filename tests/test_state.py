"""Unit tests for eosimutils.state module."""

import unittest
import random
import numpy as np

from astropy.coordinates import EarthLocation as Astropy_EarthLocation
import astropy.units as astropy_u

from eosimutils.time import AbsoluteDate
from eosimutils.frames import ReferenceFrame
from eosimutils.state import (
    Cartesian3DPosition,
    Cartesian3DVelocity,
    GeographicPosition,
    CartesianState,
)


class TestCartesian3DPosition(unittest.TestCase):
    """Test the Cartesian3DPosition class."""

    def setUp(self):
        self.x = round(random.uniform(-1e6, 1e6), 6)
        self.y = round(random.uniform(-1e6, 1e6), 6)
        self.z = round(random.uniform(-1e6, 1e6), 6)

    def test_initialization(self):
        pos = Cartesian3DPosition(
            self.x, self.y, self.z, ReferenceFrame.get("ICRF_EC")
        )
        np.testing.assert_array_equal(pos.coords, [self.x, self.y, self.z])
        self.assertEqual(pos.frame, ReferenceFrame.get("ICRF_EC"))

    def test_from_array(self):
        pos = Cartesian3DPosition.from_array(
            [self.x, self.y, self.z], ReferenceFrame.get("ITRF")
        )
        np.testing.assert_array_equal(pos.coords, [self.x, self.y, self.z])
        self.assertEqual(pos.frame, ReferenceFrame.get("ITRF"))

        pos = Cartesian3DPosition.from_array([self.x, self.y, self.z], "ITRF")
        np.testing.assert_array_equal(pos.coords, [self.x, self.y, self.z])
        self.assertEqual(pos.frame, ReferenceFrame.get("ITRF"))

        pos = Cartesian3DPosition.from_array([self.x, self.y, self.z])
        np.testing.assert_array_equal(pos.coords, [self.x, self.y, self.z])
        self.assertIsNone(pos.frame)

    def test_to_numpy(self):
        pos = Cartesian3DVelocity(
            self.x, self.y, self.z, ReferenceFrame.get("ICRF_EC")
        )
        self.assertIsInstance(pos.to_numpy(), np.ndarray)
        np.testing.assert_array_equal(
            pos.to_numpy(), np.array([self.x, self.y, self.z])
        )

    def test_to_list(self):
        pos = Cartesian3DPosition(
            self.x, self.y, self.z, ReferenceFrame.get("ICRF_EC")
        )
        self.assertEqual(pos.to_list(), [self.x, self.y, self.z])

    def test_from_dict(self):
        dict_in = {"x": self.x, "y": self.y, "z": self.z, "frame": "ICRF_EC"}
        pos = Cartesian3DPosition.from_dict(dict_in)
        np.testing.assert_array_equal(pos.coords, [self.x, self.y, self.z])
        self.assertEqual(pos.frame, ReferenceFrame.get("ICRF_EC"))

    def test_from_dict_no_frame(self):
        dict_in = {"x": self.x, "y": self.y, "z": self.z}
        pos = Cartesian3DPosition.from_dict(dict_in)
        np.testing.assert_array_equal(pos.coords, [self.x, self.y, self.z])
        self.assertIsNone(pos.frame)

    def test_to_dict(self):
        pos = Cartesian3DPosition(
            self.x, self.y, self.z, ReferenceFrame.get("ITRF")
        )
        dict_out = pos.to_dict()
        self.assertEqual(dict_out["x"], self.x)
        self.assertEqual(dict_out["y"], self.y)
        self.assertEqual(dict_out["z"], self.z)
        self.assertEqual(dict_out["frame"], "ITRF")


class TestCartesian3DVelocity(unittest.TestCase):
    """Test the Cartesian3DVelocity class."""

    def setUp(self):
        self.vx = round(random.uniform(-1e6, 1e6), 6)
        self.vy = round(random.uniform(-1e6, 1e6), 6)
        self.vz = round(random.uniform(-1e6, 1e6), 6)

    def test_initialization(self):
        vel = Cartesian3DVelocity(
            self.vx, self.vy, self.vz, ReferenceFrame.get("ICRF_EC")
        )
        np.testing.assert_array_equal(vel.coords, [self.vx, self.vy, self.vz])
        self.assertEqual(vel.frame, ReferenceFrame.get("ICRF_EC"))

    def test_from_array(self):
        vel = Cartesian3DVelocity.from_array(
            [self.vx, self.vy, self.vz], ReferenceFrame.get("ITRF")
        )
        np.testing.assert_array_equal(vel.coords, [self.vx, self.vy, self.vz])
        self.assertEqual(vel.frame, ReferenceFrame.get("ITRF"))

        vel = Cartesian3DVelocity.from_array(
            [self.vx, self.vy, self.vz], "ITRF"
        )
        np.testing.assert_array_equal(vel.coords, [self.vx, self.vy, self.vz])
        self.assertEqual(vel.frame, ReferenceFrame.get("ITRF"))

        vel = Cartesian3DVelocity.from_array([self.vx, self.vy, self.vz])
        np.testing.assert_array_equal(vel.coords, [self.vx, self.vy, self.vz])
        self.assertIsNone(vel.frame)

    def test_to_numpy(self):
        vel = Cartesian3DVelocity(
            self.vx, self.vy, self.vz, ReferenceFrame.get("ICRF_EC")
        )
        self.assertIsInstance(vel.to_numpy(), np.ndarray)
        np.testing.assert_array_equal(
            vel.to_numpy(), np.array([self.vx, self.vy, self.vz])
        )

    def test_to_list(self):
        vel = Cartesian3DVelocity(
            self.vx, self.vy, self.vz, ReferenceFrame.get("ICRF_EC")
        )
        self.assertIsInstance(vel.to_list(), list)
        self.assertEqual(vel.to_list(), [self.vx, self.vy, self.vz])

    def test_from_dict(self):
        dict_in = {
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz,
            "frame": "ICRF_EC",
        }
        vel = Cartesian3DVelocity.from_dict(dict_in)
        np.testing.assert_array_equal(vel.coords, [self.vx, self.vy, self.vz])
        self.assertEqual(vel.frame, ReferenceFrame.get("ICRF_EC"))

    def from_dict_no_frame(self):
        dict_in = {"vx": self.vx, "vy": self.vy, "vz": self.vz}
        vel = Cartesian3DVelocity.from_dict(dict_in)
        np.testing.assert_array_equal(vel.coords, [self.vx, self.vy, self.vz])
        self.assertIsNone(vel.frame)

    def test_to_dict(self):
        vel = Cartesian3DVelocity(
            self.vx, self.vy, self.vz, ReferenceFrame.get("ITRF")
        )
        dict_out = vel.to_dict()
        np.testing.assert_array_equal(vel.coords, [self.vx, self.vy, self.vz])
        self.assertEqual(dict_out["frame"], "ITRF")


class TestGeographicPosition(unittest.TestCase):
    """Test the GeographicPosition class."""

    def setUp(self):
        self.latitude_degrees = round(random.uniform(-90, 90), 6)
        self.longitude_degrees = round(random.uniform(-180, 180), 6)
        self.elevation_m = round(random.uniform(0, 10000), 6)

    def test_initialization(self):
        geo_pos = GeographicPosition(
            self.latitude_degrees, self.longitude_degrees, self.elevation_m
        )
        self.assertAlmostEqual(
            geo_pos.latitude, self.latitude_degrees, places=6
        )
        self.assertAlmostEqual(
            geo_pos.longitude, self.longitude_degrees, places=6
        )
        self.assertAlmostEqual(geo_pos.elevation, self.elevation_m, places=6)

    def test_from_dict(self):
        dict_in = {
            "latitude": self.latitude_degrees,
            "longitude": self.longitude_degrees,
            "elevation": self.elevation_m,
        }
        geo_pos = GeographicPosition.from_dict(dict_in)
        self.assertAlmostEqual(
            geo_pos.latitude, self.latitude_degrees, places=6
        )
        self.assertAlmostEqual(
            geo_pos.longitude, self.longitude_degrees, places=6
        )
        self.assertAlmostEqual(geo_pos.elevation, self.elevation_m, places=6)

    def test_to_dict(self):
        geo_pos = GeographicPosition(
            self.latitude_degrees, self.longitude_degrees, self.elevation_m
        )
        dict_out = geo_pos.to_dict()
        self.assertAlmostEqual(
            dict_out["latitude"], self.latitude_degrees, places=6
        )
        self.assertAlmostEqual(
            dict_out["longitude"], self.longitude_degrees, places=6
        )
        self.assertAlmostEqual(
            dict_out["elevation"], self.elevation_m, places=6
        )

    def test_itrs_xyz(self):
        geo_pos = GeographicPosition(37.7749, -122.4194, 10)
        itrs_xyz = geo_pos.itrs_xyz
        self.assertEqual(len(itrs_xyz), 3)
        self.assertTrue(all(isinstance(coord, float) for coord in itrs_xyz))
        # validation data from Astropy EarthLocation class
        expected_xyz = [-2706179.084e-3, -4261066.162e-3, 3885731.616e-3]
        for coord, expected in zip(itrs_xyz, expected_xyz):
            self.assertAlmostEqual(coord, expected, places=3)

    def test_itrs_xyz_astropy_validation(self):
        def geodetic_to_itrf(lat_deg: float, lon_deg: float, height_m: float):
            """
            Astropy function to convert WGS84 geodetic coordinates to
            ITRF (ECEF) Cartesian coordinates.

            Args:
                lat_deg (float): Latitude in degrees.
                lon_deg (float): Longitude in degrees.
                height_m (float): Height above WGS84 ellipsoid in meters.

            Returns:
                tuple: (x, y, z) coordinates in meters.
            """
            location = Astropy_EarthLocation.from_geodetic(
                lon=lon_deg * astropy_u.deg,
                lat=lat_deg * astropy_u.deg,
                height=height_m * astropy_u.m,
            )
            return (
                location.x.value,
                location.y.value,
                location.z.value,
            )  # Return as a tuple of floats

        geo_pos = GeographicPosition(
            self.latitude_degrees, self.longitude_degrees, self.elevation_m
        )
        itrs_xyz = geo_pos.itrs_xyz * 1e3  # convert to meters
        # validation data from Astropy EarthLocation class
        expected_xyz = geodetic_to_itrf(
            self.latitude_degrees, self.longitude_degrees, self.elevation_m
        )
        for coord, expected in zip(itrs_xyz, expected_xyz):
            self.assertAlmostEqual(coord, expected, places=3)


class TestCartesianState(unittest.TestCase):
    """Test the CartesianState class."""

    def setUp(self):
        self.time_dict = {
            "time_format": "Gregorian_Date",
            "calendar_date": "2025-03-10T14:30:00.0",
            "time_scale": "utc",
        }
        self.time = AbsoluteDate.from_dict(self.time_dict)

        self.position_dict = {
            "x": round(random.uniform(-1e6, 1e6), 6),
            "y": round(random.uniform(-1e6, 1e6), 6),
            "z": round(random.uniform(-1e6, 1e6), 6),
            "frame": "ICRF_EC",
        }
        self.position = Cartesian3DPosition.from_dict(self.position_dict)

        self.velocity_dict = {
            "vx": round(random.uniform(-1e6, 1e6), 6),
            "vy": round(random.uniform(-1e6, 1e6), 6),
            "vz": round(random.uniform(-1e6, 1e6), 6),
            "frame": "ICRF_EC",
        }
        self.velocity = Cartesian3DVelocity.from_dict(self.velocity_dict)

        self.frame = ReferenceFrame.get("ICRF_EC")

    def test_initialization(self):
        state = CartesianState(
            self.time, self.position, self.velocity, self.frame
        )
        self.assertEqual(state.time, self.time)
        np.testing.assert_array_equal(
            state.position.coords, self.position.coords
        )
        np.testing.assert_array_equal(
            state.velocity.coords, self.velocity.coords
        )
        self.assertEqual(state.frame, self.frame)

    def test_mismatched_frames(self):
        position_dict = {
            "x": round(random.uniform(-1e6, 1e6), 6),
            "y": round(random.uniform(-1e6, 1e6), 6),
            "z": round(random.uniform(-1e6, 1e6), 6),
            "frame": "ICRF_EC",
        }
        position = Cartesian3DPosition.from_dict(position_dict)

        velocity_dict = {
            "vx": round(random.uniform(-1e6, 1e6), 6),
            "vy": round(random.uniform(-1e6, 1e6), 6),
            "vz": round(random.uniform(-1e6, 1e6), 6),
            "frame": "ITRF",
        }
        velocity = Cartesian3DVelocity.from_dict(velocity_dict)

        with self.assertRaises(ValueError) as context:
            CartesianState(
                self.time, position, velocity, ReferenceFrame.get("ICRF_EC")
            )

        self.assertTrue(
            "Velocity frame does not match the provided frame."
            in str(context.exception)
        )

    def test_from_dict(self):
        dict_in = {
            "time": self.time_dict,
            "position": self.position.to_list(),
            "velocity": self.velocity.to_list(),
            "frame": "ICRF_EC",
        }
        state = CartesianState.from_dict(dict_in)
        self.assertEqual(state.time, self.time)
        np.testing.assert_array_equal(
            state.position.coords, self.position.coords
        )
        np.testing.assert_array_equal(
            state.velocity.coords, self.velocity.coords
        )
        self.assertEqual(state.frame, self.frame)

    def from_dict_no_frame(self):
        """Test from_dict method without frame."""
        dict_in = {
            "time": self.time_dict,
            "position": self.position.to_list(),
            "velocity": self.velocity.to_list(),
        }
        state = CartesianState.from_dict(dict_in)
        self.assertEqual(
            state.time.astropy_time.iso, self.time.astropy_time.iso
        )
        np.testing.assert_array_equal(
            state.position.coords, self.position.coords
        )
        np.testing.assert_array_equal(
            state.velocity.coords, self.velocity.coords
        )
        self.assertIsNone(state.frame)

    def test_to_dict(self):
        state = CartesianState(
            self.time, self.position, self.velocity, self.frame
        )
        dict_out = state.to_dict()
        self.assertEqual(dict_out["time"], self.time.to_dict())
        self.assertEqual(dict_out["position"], self.position.to_list())
        self.assertEqual(dict_out["velocity"], self.velocity.to_list())
        self.assertEqual(dict_out["frame"], "ICRF_EC")

    def test_to_skyfield_gcrf_position(self):
        # Create a CartesianState object
        state = CartesianState(
            self.time, self.position, self.velocity, self.frame
        )

        # Convert to Skyfield GCRS position
        skyfield_position = state.to_skyfield_gcrf_position()

        # Validate the Skyfield position object
        # Check that the position matches the CartesianState position
        self.assertAlmostEqual(
            skyfield_position.position.km[0], self.position.coords[0], places=6
        )
        self.assertAlmostEqual(
            skyfield_position.position.km[1], self.position.coords[1], places=6
        )
        self.assertAlmostEqual(
            skyfield_position.position.km[2], self.position.coords[2], places=6
        )

        # Check that the velocity matches the CartesianState velocity
        self.assertAlmostEqual(
            skyfield_position.velocity.km_per_s[0],
            self.velocity.coords[0],
            places=6,
        )
        self.assertAlmostEqual(
            skyfield_position.velocity.km_per_s[1],
            self.velocity.coords[1],
            places=6,
        )
        self.assertAlmostEqual(
            skyfield_position.velocity.km_per_s[2],
            self.velocity.coords[2],
            places=6,
        )

        # Check that the time matches the CartesianState time
        self.assertEqual(
            skyfield_position.t.utc_iso(),
            "2025-03-10T14:30:00Z",
        )

    def test_to_skyfield_gcrf_position_invalid_frame(self):
        """Test that ValueError is raised when frame is not ICRF_EC."""
        # Create a CartesianState object with a non-ICRF_EC frame
        position = Cartesian3DPosition(
            self.position.coords[0],
            self.position.coords[1],
            self.position.coords[2],
            ReferenceFrame.get("ITRF"),
        )
        velocity = Cartesian3DVelocity(
            self.velocity.coords[0],
            self.velocity.coords[1],
            self.velocity.coords[2],
            ReferenceFrame.get("ITRF"),
        )
        state = CartesianState(
            self.time, position, velocity, ReferenceFrame.get("ITRF")
        )

        # Check that ValueError is raised
        with self.assertRaises(ValueError) as context:
            state.to_skyfield_gcrf_position()

        self.assertTrue(
            "Only CartesianState object in ICRF_EC frame is supported for "
            "conversion to Skyfield GCRF position." in str(context.exception)
        )

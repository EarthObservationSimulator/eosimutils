"""Unit tests for eosimutils.base module."""

import unittest


from eosimutils.base import EnumBase, ReferenceFrame


class TestReferenceFrame(unittest.TestCase):
    """Test the ReferenceFrame enum."""

    def test_enum_values(self):
        # Test that all enum values are uppercase
        for frame in ReferenceFrame:
            self.assertTrue(frame.value.isupper())

    def test_get(self):
        # Test valid input
        self.assertEqual(ReferenceFrame.get("EARTH_ICRF"), ReferenceFrame.EARTH_ICRF)
        self.assertEqual(ReferenceFrame.get("earth_ICrf"), ReferenceFrame.EARTH_ICRF)
        self.assertEqual(
            ReferenceFrame.get(["EARTH_ICRF", "ITRF"]),
            [ReferenceFrame.EARTH_ICRF, ReferenceFrame.ITRF],
        )

        # Test invalid input
        self.assertIsNone(ReferenceFrame.get("INVALID"))
        self.assertIsNone(ReferenceFrame.get(["EARTH_ICRF", "INVALID"])[1])

    def test_to_string(self):
        # Test string representation
        self.assertEqual(ReferenceFrame.EARTH_ICRF.to_string(), "EARTH_ICRF")
        self.assertEqual(ReferenceFrame.ITRF.to_string(), "ITRF")

    def test_equality(self):
        # Test equality with EnumBase
        self.assertEqual(ReferenceFrame.EARTH_ICRF, ReferenceFrame.EARTH_ICRF)
        self.assertNotEqual(ReferenceFrame.EARTH_ICRF, ReferenceFrame.ITRF)

        # Test equality with string
        self.assertEqual(ReferenceFrame.EARTH_ICRF, "EARTH_ICRF")
        self.assertNotEqual(ReferenceFrame.EARTH_ICRF, "ITRF")

        # Test equality with other types
        self.assertNotEqual(ReferenceFrame.EARTH_ICRF, 123)
        self.assertNotEqual(ReferenceFrame.EARTH_ICRF, None)

"""Unit tests for eosimutils.base module."""

import unittest


from eosimutils.base import ReferenceFrame


class TestReferenceFrame(unittest.TestCase):
    """Test the ReferenceFrame class."""

    def test_values_uppercase(self):
        # Test that all values are uppercase
        for frame in ReferenceFrame.values():
            self.assertTrue(frame.to_string().isupper())

    def test_get(self):
        # Test valid input
        self.assertEqual(ReferenceFrame.get("ICRF_EC"), "ICRF_EC")
        # Test invalid input
        self.assertIsNone(ReferenceFrame.get("INVALID"))

    def test_to_string(self):
        # Test string representation
        self.assertEqual(ReferenceFrame.get("ICRF_EC").to_string(), "ICRF_EC")
        self.assertEqual(ReferenceFrame.get("ITRF").to_string(), "ITRF")

    def test_equality(self):
        # Test equality with string
        self.assertEqual(ReferenceFrame.get("ICRF_EC"), "ICRF_EC")
        self.assertNotEqual(ReferenceFrame.get("ICRF_EC"), "ITRF")

        # Test equality with other types
        self.assertNotEqual(ReferenceFrame.get("ICRF_EC"), 123)
        self.assertNotEqual(ReferenceFrame.get("ICRF_EC"), None)

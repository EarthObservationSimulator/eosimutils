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
        self.assertEqual(ReferenceFrame.get("GCRF"), ReferenceFrame.GCRF)
        self.assertEqual(ReferenceFrame.get("gcrf"), ReferenceFrame.GCRF)
        self.assertEqual(
            ReferenceFrame.get(["GCRF", "ITRF"]),
            [ReferenceFrame.GCRF, ReferenceFrame.ITRF],
        )

        # Test invalid input
        self.assertIsNone(ReferenceFrame.get("INVALID"))
        self.assertIsNone(ReferenceFrame.get(["GCRF", "INVALID"])[1])

    def test_to_string(self):
        # Test string representation
        self.assertEqual(ReferenceFrame.GCRF.to_string(), "GCRF")
        self.assertEqual(ReferenceFrame.ITRF.to_string(), "ITRF")

    def test_equality(self):
        # Test equality with EnumBase
        self.assertEqual(ReferenceFrame.GCRF, ReferenceFrame.GCRF)
        self.assertNotEqual(ReferenceFrame.GCRF, ReferenceFrame.ITRF)

        # Test equality with string
        self.assertEqual(ReferenceFrame.GCRF, "GCRF")
        self.assertNotEqual(ReferenceFrame.GCRF, "ITRF")

        # Test equality with other types
        self.assertNotEqual(ReferenceFrame.GCRF, 123)
        self.assertNotEqual(ReferenceFrame.GCRF, None)

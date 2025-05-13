"""Unit tests for eosimutils.base module."""

import unittest


from eosimutils.frames import ReferenceFrame


class TestReferenceFrame(unittest.TestCase):
    """Test the ReferenceFrame enum."""

    def test_enum_values(self):
        # Test that all enum values are uppercase
        for frame in ReferenceFrame:
            self.assertTrue(frame.to_string().isupper())

    def test_get(self):
        # Test valid input
        self.assertEqual(ReferenceFrame.get("ICRF_EC"), ReferenceFrame.ICRF_EC)
        self.assertEqual(ReferenceFrame.get("ICRF_EC"), ReferenceFrame.ICRF_EC)
        self.assertEqual(
            ReferenceFrame.get(["ICRF_EC", "ITRF"]),
            [ReferenceFrame.ICRF_EC, ReferenceFrame.ITRF],
        )

        # Test invalid input
        self.assertIsNone(ReferenceFrame.get("INVALID"))
        self.assertIsNone(ReferenceFrame.get(["ICRF_EC", "INVALID"])[1])

    def test_to_string(self):
        # Test string representation
        self.assertEqual(ReferenceFrame.ICRF_EC.to_string(), "ICRF_EC")
        self.assertEqual(ReferenceFrame.ITRF.to_string(), "ITRF")

    def test_equality(self):
        # Test equality with EnumBase
        self.assertEqual(ReferenceFrame.ICRF_EC, ReferenceFrame.ICRF_EC)
        self.assertNotEqual(ReferenceFrame.ICRF_EC, ReferenceFrame.ITRF)

        # Test equality with string
        self.assertEqual(ReferenceFrame.ICRF_EC, "ICRF_EC")
        self.assertNotEqual(ReferenceFrame.ICRF_EC, "ITRF")

        # Test equality with other types
        self.assertNotEqual(ReferenceFrame.ICRF_EC, 123)
        self.assertNotEqual(ReferenceFrame.ICRF_EC, None)

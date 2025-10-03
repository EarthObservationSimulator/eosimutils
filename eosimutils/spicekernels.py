"""
.. module:: eosimutils.spicekernels
   :synopsis: Module for handling SPICE kernels.

Module for handling SPICE kernels.
"""

import spiceypy as spice
import os
import urllib.request

lsk_kernel_file_name = "naif0012.tls"
eop_kernel_file_name = "earth_1962_250826_2125_combined.bpc"
de430_kernel_file_name = "de430.bsp"

def download_latest_kernels() -> None:
    """Download the latest SPICE kernels from the NAIF website.

    This function downloads the latest SPICE kernels from the NAIF website
    and stores them in a local directory. The directory is created if it
    does not exist.

    Raises:
       FileNotFoundError: If the SPICE kernel files are not found.
    """
    # Define the directory to save kernels
    kernel_dir = os.path.join(os.path.dirname(__file__), "spice_kernels")
    os.makedirs(
        kernel_dir, exist_ok=True
    )  # do not create the directory if it already exists

    # Define the latest kernel URLs
    kernels = {
        "LSK": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/"+lsk_kernel_file_name,  # pylint: disable=line-too-long
        "BPC": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/"+eop_kernel_file_name,  # pylint: disable=line-too-long
        "DE430": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"+de430_kernel_file_name,  # pylint: disable=line-too-long
    }

    # Function to download a kernel if it doesn't already exist
    def download_kernel(url):
        filename = os.path.join(kernel_dir, os.path.basename(url))

        if os.path.exists(filename):
            pass
        else:
            try:
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filename)
                print(f"Downloaded and saved: {filename}\n")
            except Exception as e:  # pylint: disable=broad-exception-caught
                print(f"Failed to download {filename}: {e}")

    # Download the required kernels
    for url in kernels.values():
        download_kernel(url)


def load_spice_kernels() -> None:
    """Load SPICE kernel files required for time conversions.

    Raises:
       FileNotFoundError: If the SPICE kernel files are not found.
    """
    # Download the kernels if they are not already present
    download_latest_kernels()

    # Load the kernels
    kernel_dir = os.path.join(os.path.dirname(__file__), "spice_kernels")
    leap_seconds_kernel = os.path.join(kernel_dir, lsk_kernel_file_name)
    eop_kernel = os.path.join(kernel_dir, eop_kernel_file_name)
    de430_kernel = os.path.join(
        kernel_dir, de430_kernel_file_name
    )  # contains ephemeris data of planets, Moon, Sun, Pluto

    try:
        spice.unload(leap_seconds_kernel)  # Unload if already loaded
        spice.furnsh(leap_seconds_kernel)  # Load Leap Seconds Kernel

        spice.unload(eop_kernel)  # Unload if already loaded
        spice.furnsh(eop_kernel)  # Load High-precision EOP Kernel

        spice.unload(de430_kernel)  # Unload if already loaded
        spice.furnsh(de430_kernel)  # Load DE430 Ephemeris Kernel
    except spice.utils.exceptions.SpiceyError as e:
        raise FileNotFoundError(
            f"SPICE kernel files not found. Please check the paths:\n"
            f"Leap Seconds Kernel: {leap_seconds_kernel}\n"
            f"EOP Kernel: {eop_kernel}\n"
            f"DE430 Kernel: {de430_kernel}"
        ) from e

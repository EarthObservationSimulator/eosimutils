"""
.. module:: eosimutils.spicekernels
   :synopsis: Module for handling SPICE kernels.

Module for handling SPICE kernels.
"""

import spiceypy as spice
import os
import urllib.request

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
   os.makedirs(kernel_dir, exist_ok=True) # do not create the directory if it already exists

   # Define the latest kernel URLs
   kernels = {
      "LSK": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
      "BPC": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_000101_250627_250331.bpc",
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
         except Exception as e:
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
   leap_seconds_kernel = os.path.join(kernel_dir, "naif0012.tls")
   eop_kernel = os.path.join(kernel_dir, "earth_000101_250627_250331.bpc")

   try:
      spice.furnsh(leap_seconds_kernel)  # Load Leap Seconds Kernel
      spice.furnsh(eop_kernel)  # Load High-precision EOP Kernel
   except spice.utils.exceptions.SpiceyError as e:
      raise FileNotFoundError(
            f"SPICE kernel files not found. Please check the paths:\n"
            f"Leap Seconds Kernel: {leap_seconds_kernel}\n"
            f"EOP Kernel: {eop_kernel}"
      ) from e
"""Tests for the public-benchmark adapters.

Everything here runs offline against fixtures constructed in the test files.
No test in this directory may download a dataset or read one from a user's
cache, because the property being protected is that the suite passes on a
machine with nothing on it and no API key.
"""

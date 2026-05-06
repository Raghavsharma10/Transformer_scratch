def _prep_bins():
    """
    Support for running straight out of a cloned source directory instead
    of an installed distribution
    """

    from os import path
    from sys import platform, maxsize
    from shutil import copy
    bit_suffix = "-x86_64" if maxsize > 2**32 else "-x86"
    package_root = path.abspath(path.dirname(__file__))
    prebuilt_path = path.join(package_root, "prebuilt", platform + bit_suffix)
    config = {"MANIFEST_DIR": prebuilt_path}
    try:
        execfile(path.join(prebuilt_path, "manifest.pycfg"), config)
    except IOError:
        return  # there are no prebuilts for this platform - nothing to do
    files = map(lambda x: path.join(prebuilt_path, x), config["FILES"])
    for prebuilt_file in files:
        try:
            copy(path.join(prebuilt_path, prebuilt_file), package_root)
        except IOError:
            pass
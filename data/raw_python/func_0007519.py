def main(argv, version=DEFAULT_VERSION):
    """Install or upgrade setuptools and EasyInstall"""
    tarball = download_setuptools()
    _install(tarball, _build_install_args(argv))
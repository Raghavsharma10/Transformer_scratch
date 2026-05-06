def vendorize(
    c,
    distribution,
    version,
    vendor_dir,
    package=None,
    git_url=None,
    license=None,
):
    """
    Vendorize Python package ``distribution`` at version/SHA ``version``.

    Specify the vendor folder (e.g. ``<mypackage>/vendor``) as ``vendor_dir``.

    For Crate/PyPI releases, ``package`` should be the name of the software
    entry on those sites, and ``version`` should be a specific version number.
    E.g. ``vendorize('lexicon', '0.1.2')``.

    For Git releases, ``package`` should be the name of the package folder
    within the checkout that needs to be vendorized and ``version`` should be a
    Git identifier (branch, tag, SHA etc.) ``git_url`` must also be given,
    something suitable for ``git clone <git_url>``.

    For SVN releases: xxx.

    For packages where the distribution name is not the same as the package
    directory name, give ``package='name'``.

    By default, no explicit license seeking is done -- we assume the license
    info is in file headers or otherwise within the Python package vendorized.
    This is not always true; specify ``license=/path/to/license/file`` to
    trigger copying of a license into the vendored folder from the
    checkout/download (relative to its root.)
    """
    with tmpdir() as tmp:
        package = package or distribution
        target = os.path.join(vendor_dir, package)
        # Unpack source
        real_version, source = _unpack(c, tmp, distribution, version, git_url)
        abs_source = os.path.join(tmp, source)
        source_package = os.path.join(abs_source, package)
        # Ensure source package exists
        if not os.path.exists(source_package):
            rel_package = os.path.join(source, package)
            raise ValueError("Source package %s doesn't exist!" % rel_package)
        # Nuke target if exists
        if os.path.exists(target):
            print("Removing pre-existing vendorized folder %s" % target)
            rmtree(target)
        # Perform the copy
        print("Copying %s => %s" % (source_package, target))
        copytree(source_package, target)
        # Explicit license if needed
        if license:
            copy(os.path.join(abs_source, license), target)
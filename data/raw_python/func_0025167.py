def multiple_packaged_versions(package_name):
    """ Look through built package directory and see if there are multiple versions there """
    dist_files = os.listdir('dist')
    versions = set()
    for filename in dist_files:
        version = funcy.re_find(r'{}-(.+).tar.gz'.format(package_name), filename)
        if version:
            versions.add(version)
    return len(versions) > 1
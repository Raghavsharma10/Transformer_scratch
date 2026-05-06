def build_package(path, requires, virtualenv=None, ignore=None,
                  extra_files=None, zipfile_name=ZIPFILE_NAME,
                  pyexec=None):
    '''Builds the zip file and creates the package with it'''
    pkg = Package(path, zipfile_name, pyexec)

    if extra_files:
        for fil in extra_files:
            pkg.extra_file(fil)
    if virtualenv is not None:
        pkg.virtualenv(virtualenv)
    pkg.requirements(requires)
    pkg.build(ignore)

    return pkg
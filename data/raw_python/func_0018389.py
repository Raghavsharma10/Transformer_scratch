def _version(self):
        """Deduce the version number of the downloaded package from its filename."""
        # TODO: Can we delete this method and just print the line from the
        # reqs file verbatim instead?
        def version_of_archive(filename, package_name):
            # Since we know the project_name, we can strip that off the left, strip
            # any archive extensions off the right, and take the rest as the
            # version.
            for ext in ARCHIVE_EXTENSIONS:
                if filename.endswith(ext):
                    filename = filename[:-len(ext)]
                    break
            # Handle github sha tarball downloads.
            if is_git_sha(filename):
                filename = package_name + '-' + filename
            if not filename.lower().replace('_', '-').startswith(package_name.lower()):
                # TODO: Should we replace runs of [^a-zA-Z0-9.], not just _, with -?
                give_up(filename, package_name)
            return filename[len(package_name) + 1:]  # Strip off '-' before version.

        def version_of_wheel(filename, package_name):
            # For Wheel files (http://legacy.python.org/dev/peps/pep-0427/#file-
            # name-convention) we know the format bits are '-' separated.
            whl_package_name, version, _rest = filename.split('-', 2)
            # Do the alteration to package_name from PEP 427:
            our_package_name = re.sub(r'[^\w\d.]+', '_', package_name, re.UNICODE)
            if whl_package_name != our_package_name:
                give_up(filename, whl_package_name)
            return version

        def give_up(filename, package_name):
            raise RuntimeError("The archive '%s' didn't start with the package name "
                               "'%s', so I couldn't figure out the version number. "
                               "My bad; improve me." %
                               (filename, package_name))

        get_version = (version_of_wheel
                       if self._downloaded_filename().endswith('.whl')
                       else version_of_archive)
        return get_version(self._downloaded_filename(), self._project_name())
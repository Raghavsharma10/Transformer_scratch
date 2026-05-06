def package(self, ignore=None):
        """
        Create a zip file of the lambda script and its dependencies.

        :param list ignore: a list of regular expression strings to match paths
            of files in the source of the lambda script against and ignore
            those files when creating the zip file. The paths to be matched are
            local to the source root.
        """
        ignore = ignore or []
        package = os.path.join(self._temp_workspace, 'lambda_package')

        # Copy site packages into package base
        LOG.info('Copying site packages')

        if hasattr(self, '_pkg_venv') and self._pkg_venv:
            lib_dir = 'lib/python*/site-packages'
            lib64_dir = 'lib64/python*/site-packages'

            if sys.platform == 'win32' or sys.platform == 'cygwin':
                lib_dir = 'lib\\site-packages'
                lib64_dir = 'lib64\\site-packages'

            # Look for the site packages
            lib_site_list = glob.glob(os.path.join(
                self._pkg_venv, lib_dir))
            if lib_site_list:
                utils.copy_tree(lib_site_list[0], package)
            else:
                LOG.debug("no lib site packages found")

            lib64_site_list = glob.glob(os.path.join(
                self._pkg_venv, lib64_dir))
            if lib64_site_list:
                lib64_site_packages = lib64_site_list[0]
                if not os.path.islink(lib64_site_packages):
                    LOG.info('Copying lib64 site packages')
                    utils.copy_tree(lib64_site_packages, package)
                lib64_site_packages = lib64_site_list[0]
            else:
                LOG.debug("no lib64 site packages found")

        # Append the temp workspace to the ignore list:
        ignore.append(r"^%s/.*" % re.escape(TEMP_WORKSPACE_NAME))
        utils.copy_tree(self._path, package, ignore)

        # Add extra files
        for p in self._extra_files:
            LOG.info('Copying extra %s into package' % p)
            ignore.append(re.escape(p))
            if os.path.isdir(p):
                utils.copy_tree(p, package, ignore=ignore, include_parent=True)
            else:
                shutil.copy(p, package)

        self._create_zip(package)
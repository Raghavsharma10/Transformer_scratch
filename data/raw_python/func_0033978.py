def pip_install_to_target(self, path, requirements="", local_package=None):
        """For a given active virtualenv, gather all installed pip packages then
        copy (re-install) them to the path provided.
        :param str path:
            Path to copy installed pip packages to.
        :param str requirements:
            If set, only the packages in the requirements.txt file are installed.
            The requirements.txt file needs to be in the same directory as the
            project which shall be deployed.
            Defaults to false and installs all pacakges found via pip freeze if
            not set.
        :param str local_package:
            The path to a local package with should be included in the deploy as
            well (and/or is not available on PyPi)
        """
        packages = []
        if not requirements:
            logger.debug('Gathering pip packages')
            # packages.extend(pip.operations.freeze.freeze())
            pass
        else:
            requirements_path = os.path.join(self.get_src_path(), requirements)
            logger.debug('Gathering packages from requirements: {}'.format(requirements_path))
            if os.path.isfile(requirements_path):
                data = self.read(requirements_path)
                packages.extend(data.splitlines())
            else:
                logger.debug('No requirements file in {}'.format(requirements_path))

        if local_package is not None:
            if not isinstance(local_package, (list, tuple)):
                local_package = [local_package]
            for l_package in local_package:
                packages.append(l_package)
        self._install_packages(path, packages)
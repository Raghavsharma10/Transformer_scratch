def _install_packages(self, path, packages):
        """Install all packages listed to the target directory.
        Ignores any package that includes Python itself and python-lambda as well
        since its only needed for deploying and not running the code
        :param str path:
            Path to copy installed pip packages to.
        :param list packages:
            A list of packages to be installed via pip.
        """

        def _filter_blacklist(package):
            blacklist = ["-i", "#", "Python==", "ardy=="]
            return all(package.startswith(entry.encode()) is False for entry in blacklist)

        filtered_packages = filter(_filter_blacklist, packages)
        # print([package for package in filtered_packages])
        for package in filtered_packages:
            package = str(package, "utf-8")
            if package.startswith('-e '):
                package = package.replace('-e ', '')

            logger.info('Installing {package}'.format(package=package))
            pip.main(['install', package, '-t', path, '--ignore-installed', '-q'])
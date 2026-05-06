def _installer(self, package_list, install_string=None):
        """Install operating system packages for the system.

        :param: package_list: ``list``
        :param install_string: ``str``
        """
        packages = ' '.join(package_list)

        if install_string is None:
            self.install_string = self.install_process[self.distro] % packages
        else:
            self.install_string = install_string

        output, outcome = self.shell.run_command(command=self.install_string)

        if outcome is False:
            raise IOError(output)
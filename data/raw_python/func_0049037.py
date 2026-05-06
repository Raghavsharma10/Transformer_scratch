def install(self):
        """Install packages from the packages_dict."""
        self.distro = distro_check()
        package_list = self.packages_dict.get(self.distro)
        self._installer(package_list=package_list.get('packages'))
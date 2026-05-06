def add_package(self, package):
        """
        Add a package to this project
        """
        self._data.setdefault('packages', {})
        
        self._data['packages'][package.name] = package.source

        for package in package.deploy_packages:
            self.add_package(package)

        self._save()
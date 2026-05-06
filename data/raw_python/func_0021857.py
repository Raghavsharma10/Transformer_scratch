def package_manager_owns(self, dist):
        """
        Returns True if package manager 'owns' file
        Returns False if package manager does not 'own' file

        There is currently no way to determine if distutils or
        setuptools installed a package. A future feature of setuptools
        will make a package manifest which can be checked.
           
        'filename' must be the full path to file

        """
        #Installed by distutils/setuptools or external package manager?
        #If location is in site-packages dir, check for .egg-info file
        if dist.location.lower() == get_python_lib().lower():
            filename = os.path.join(dist.location, dist.egg_name() + ".egg-info")
        else:
            filename = dist.location

        status, output = getstatusoutput("/usr/bin/acmefile -q %s" % filename)
        #status == 0 (file was installed by Acme)
        #status == 256 (file was not installed by Acme)
        if status == 0:
            return self.name
        else:
            return ""
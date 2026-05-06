def _process_dependency(self, dep_t, dep_l):
        """Add dependencies into self.dependencies, possibly also adding system packages
        that contain non-distro package managers (e.g. if someone wants to install
        dependencies with pip and pip is not present, it will get installed through
        RPM on RPM based systems, etc.

        Skips dependencies that are supposed to be installed by system manager that
        is not native to this system.
        """
        if dep_t not in managers:
            err = 'No package manager for dependency type "{dep_t}"'.format(dep_t=dep_t)
            raise exceptions.NoPackageManagerException(err)
        # try to get list of distros where the dependency type is system type
        distros = settings.SYSTEM_DEPTYPES_SHORTCUTS.get(dep_t, None)
        if not distros:  # non-distro dependency type
            sysdep_t = self.get_system_deptype_shortcut()
            # for now, just take the first manager that can install dep_t and install this manager
            self._process_dependency(sysdep_t,
                                     managers[dep_t][0].get_distro_dependencies(sysdep_t))
        else:
            local_distro = utils.get_distro_name()
            found = False
            for distro in distros:
                if distro in local_distro:
                    found = True
                    break
            if not found:  # distro dependency type, but for another distro
                return
        self.__add_dependencies(dep_t, dep_l)
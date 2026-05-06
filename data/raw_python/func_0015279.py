def get_package_manager(self, dep_t):
        """Choose proper package manager and return it."""
        mgrs = managers.get(dep_t, [])
        for manager in mgrs:
            if manager.works():
                return manager
        if not mgrs:
            err = 'No package manager for dependency type "{dep_t}"'.format(dep_t=dep_t)
            raise exceptions.NoPackageManagerException(err)
        else:
            mgrs_nice = ', '.join([mgr.__name__ for mgr in mgrs])
            err = 'No working package manager for "{dep_t}" in: {mgrs}'.format(dep_t=dep_t,
                                                                              mgrs=mgrs_nice)
            raise exceptions.NoPackageManagerOperationalException(err)
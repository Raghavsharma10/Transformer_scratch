def get_manager(self, osid=None, impl_class_name=None, version=None):
        """Finds, loads and instantiates providers of OSID managers.

        Providers must conform to an OsidManager interface. The
        interfaces are defined in the OSID enumeration. For all OSID
        requests, an instance of ``OsidManager`` that implements the
        ``OsidManager`` interface is returned. In bindings where
        permitted, this can be safely cast into the requested manager.

        arg:    osid (osid.OSID): represents the OSID
        arg:    impl_class_name (string): the name of the implementation
        arg:    version (osid.installation.Version): the minimum
                required OSID specification version
        return: (osid.OsidManager) - the manager of the service
        raise:  ConfigurationError - an error in configuring the
                implementation
        raise:  NotFound - the implementation class was not found
        raise:  NullArgument - ``impl_class_name`` or ``version`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unsupported - ``impl_class_name`` does not support the
                requested OSID
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: After finding and instantiating the
        requested ``OsidManager,`` providers must invoke
        ``OsidManager.initialize(OsidRuntimeManager)`` where the
        environment is an instance of the current environment that
        includes the configuration for the service being initialized.
        The ``OsidRuntimeManager`` passed may include information useful
        for the configuration such as the identity of the service being
        instantiated.

        """
        # This implementation assumes that all osid impls reside as seperate
        # packages in the dlkit library, so that for instance the proxy manager for an
        # OSID = 'osidpackage' in an implementation named 'impl_name' manager can
        # be found in the python path for the module: dlkit.impl_name.osid.managers
        # Also this implementation currently ignores the OSID specification version.
        from importlib import import_module
        try:
            manager_module = import_module('dlkit.' + impl_class_name + '.' + osid.lower() + '.managers')
        except ImportError:
            raise NotFound()
        try:
            manager = getattr(manager_module, osid.title() + 'Manager')
        except AttributeError:
            raise Unsupported()
        return manager
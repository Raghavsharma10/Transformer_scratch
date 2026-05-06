def initialize(self, runtime=None):
        """Initializes this manager.
        A manager is initialized once at the time of creation.
        arg:    runtime (osid.OsidRuntimeManager): the runtime
                environment
        raise:  CONFIGURATION_ERROR - an error with implementation
                configuration
        raise:  ILLEGAL_STATE - this manager has already been
                initialized by the OsidRuntime
        raise:  NullArgument - runtime is null
        raise:  OperationFailed - unable to complete request
        compliance: mandatory - This method must be implemented.
        implementation notes: In addition to loading its runtime
        configuration an implementation may create shared resources such
        as connection pools to be shared among all sessions of this
        service and released when this manager is closed. Providers must
        thread-protect any data stored in the manager.  To maximize
        interoperability, providers should not honor a second call to
        initialize() and must set an ILLEGAL_STATE error.

        """
        if self._runtime is not None:
            raise IllegalState()
        self._runtime = runtime
        config = runtime.get_configuration()
        parameter_id = Id('parameter:hostName@dlkit_service')
        host = config.get_value_by_parameter(parameter_id).get_string_value()
        if host is not None:
            self._host = host
        parameter_id = Id('parameter:appKey@dlkit_service')
        app_key = config.get_value_by_parameter(parameter_id).get_string_value()
        if app_key is not None:
            self._app_key = app_key
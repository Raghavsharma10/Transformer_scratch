def _init_catalog(self, proxy=None, runtime=None):
        """Initialize this session as an OsidCatalog based session."""
        self._init_proxy_and_runtime(proxy, runtime)
        osid_name = self._session_namespace.split('.')[0]
        try:
            config = self._runtime.get_configuration()
            parameter_id = Id('parameter:' + osid_name + 'CatalogingProviderImpl@mongo')
            provider_impl = config.get_value_by_parameter(parameter_id).get_string_value()
            self._cataloging_manager = self._runtime.get_manager('CATALOGING', provider_impl)  # need to add version argument
        except (AttributeError, KeyError, errors.NotFound):
            pass
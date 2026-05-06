def _validate_resource_desc(self):
        """
        **Purpose**: Validate the resource description provided to the ResourceManager
        """

        self._prof.prof('validating rdesc', uid=self._uid)
        self._logger.debug('Validating resource description')

        expected_keys = ['resource',
                         'walltime',
                         'cpus']

        for key in expected_keys:
            if key not in self._resource_desc:
                raise MissingError(obj='resource description', missing_attribute=key)

        if not isinstance(self._resource_desc['resource'], str):
            raise TypeError(expected_type=str, actual_type=type(self._resource_desc['resource']))

        if not isinstance(self._resource_desc['walltime'], int):
            raise TypeError(expected_type=int, actual_type=type(self._resource_desc['walltime']))

        if not isinstance(self._resource_desc['cpus'], int):
            raise TypeError(expected_type=int, actual_type=type(self._resource_desc['cpus']))

        if 'gpus' in self._resource_desc:
            if (not isinstance(self._resource_desc['gpus'], int)):
                raise TypeError(expected_type=int, actual_type=type(self._resource_desc['project']))

        if 'project' in self._resource_desc:
            if (not isinstance(self._resource_desc['project'], str)) and (not self._resource_desc['project']):
                raise TypeError(expected_type=str, actual_type=type(self._resource_desc['project']))

        if 'access_schema' in self._resource_desc:
            if not isinstance(self._resource_desc['access_schema'], str):
                raise TypeError(expected_type=str, actual_type=type(self._resource_desc['access_schema']))

        if 'queue' in self._resource_desc:
            if not isinstance(self._resource_desc['queue'], str):
                raise TypeError(expected_type=str, actual_type=type(self._resource_desc['queue']))

        if not isinstance(self._rts_config, dict):
            raise TypeError(expected_type=dict, actual_type=type(self._rts_config))

        self._validated = True

        self._logger.info('Resource description validated')
        self._prof.prof('rdesc validated', uid=self._uid)

        return self._validated
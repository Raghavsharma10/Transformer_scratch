def _populate(self):
        """
        **Purpose**:    Populate the ResourceManager class with the validated
                        resource description
        """

        if self._validated:

            self._prof.prof('populating rmgr', uid=self._uid)
            self._logger.debug('Populating resource manager object')

            self._resource = self._resource_desc['resource']
            self._walltime = self._resource_desc['walltime']
            self._cpus = self._resource_desc['cpus']
            self._gpus = self._resource_desc.get('gpus', 0)
            self._project = self._resource_desc.get('project', None)
            self._access_schema = self._resource_desc.get('access_schema', None)
            self._queue = self._resource_desc.get('queue', None)

            self._logger.debug('Resource manager population successful')
            self._prof.prof('rmgr populated', uid=self._uid)

        else:
            raise EnTKError('Resource description not validated')
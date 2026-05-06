def import_module(self, modules, shared=False, into_spooler=False):
        """Imports a python module.

        :param list|str|unicode modules:

        :param bool shared: Import a python module in all of the processes.
            This is done after fork but before request processing.

        :param bool into_spooler: Import a python module in the spooler.
            http://uwsgi-docs.readthedocs.io/en/latest/Spooler.html

        """
        if all((shared, into_spooler)):
            raise ConfigurationError('Unable to set both `shared` and `into_spooler` flags')

        if into_spooler:
            command = 'spooler-python-import'
        else:
            command = 'shared-python-import' if shared else 'python-import'

        self._set(command, modules, multi=True)

        return self._section
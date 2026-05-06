def register_module_alias(self, alias, module_path, after_init=False):
        """Adds an alias for a module.

        http://uwsgi-docs.readthedocs.io/en/latest/PythonModuleAlias.html

        :param str|unicode alias:
        :param str|unicode module_path:
        :param bool after_init: add a python module alias after uwsgi module initialization
        """
        command = 'post-pymodule-alias' if after_init else 'pymodule-alias'
        self._set(command, '%s=%s' % (alias, module_path), multi=True)

        return self._section
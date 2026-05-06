def _get_config(self, i):
        """
        Get the config.

        :type i: cleo.inputs.input.Input

        :rtype: dict
        """
        variables = {}
        if not i.get_option('config'):
            raise Exception('The --config|-c option is missing.')

        with open(i.get_option('config')) as fh:
            exec(fh.read(), {}, variables)

        return variables['DATABASES']
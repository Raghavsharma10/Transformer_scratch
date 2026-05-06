def number(self, assignment_class=None, namespace='d'):
        """
        Return a new number.

        :param assignment_class: Determines the length of the number. Possible values are 'authority' (3 characters) ,
            'registered' (5) , 'unregistered' (7)  and 'self' (9). Self assigned numbers are random and acquired locally,
            while the other assignment classes use the number server defined in the configuration. If None,
            then look in the number server configuration for one of the class keys, starting
            with the longest class and working to the shortest.
        :param namespace: The namespace character, the first character in the number. Can be one of 'd', 'x' or 'b'
        :return:
        """
        if assignment_class == 'self':
            # When 'self' is explicit, don't look for number server config
            return str(DatasetNumber())

        elif assignment_class is None:

            try:
                nsconfig = self.services['numbers']

            except ConfigurationError:
                # A missing configuration is equivalent to 'self'
                self.logger.error('No number server configuration; returning self assigned number')
                return str(DatasetNumber())

            for assignment_class in ('self', 'unregistered', 'registered', 'authority'):
                if assignment_class+'-key' in nsconfig:
                    break

            # For the case where the number configuratoin references a self-assigned key
            if assignment_class == 'self':
                return str(DatasetNumber())

        else:
            try:
                nsconfig = self.services['numbers']

            except ConfigurationError:
                raise ConfigurationError('No number server configuration')

            if assignment_class + '-key' not in nsconfig:
                raise ConfigurationError(
                    'Assignment class {} not number server config'.format(assignment_class))

        try:

            key = nsconfig[assignment_class + '-key']
            config = {
                'key': key,
                'host': nsconfig['host'],
                'port': nsconfig.get('port', 80)
            }

            ns = NumberServer(**config)

            n = str(next(ns))
            self.logger.info('Got number from number server: {}'.format(n))

        except HTTPError as e:
            self.logger.error('Failed to get number from number server for key: {}'.format(key, e.message))
            self.logger.error('Using self-generated number. There is no problem with this, '
                              'but they are longer than centrally generated numbers.')
            n = str(DatasetNumber())

        return n
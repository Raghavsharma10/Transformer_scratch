def generate_PID_name(self, prefix=None):
        '''
        Generate a unique random Handle name (random UUID). The Handle is not
        registered. If a prefix is specified, the PID name has the syntax
        <prefix>/<generatedname>, otherwise it just returns the generated
        random name (suffix for the Handle).

        :param prefix: Optional. The prefix to be used for the Handle name.
        :return: The handle name in the form <prefix>/<generatedsuffix> or
            <generatedsuffix>.
        '''

        LOGGER.debug('generate_PID_name...')

        randomuuid = uuid.uuid4()
        if prefix is not None:
            return prefix + '/' + str(randomuuid)
        else:
            return str(randomuuid)
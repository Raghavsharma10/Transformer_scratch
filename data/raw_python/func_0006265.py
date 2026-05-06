def add_auth_attempt(self, auth_type, successful, **kwargs):
        """
        :param username:
        :param password:
        :param auth_type: possible values:
                                plain: plaintext username/password
        :return:
        """

        entry = {'timestamp': datetime.utcnow(),
                 'auth': auth_type,
                 'id': uuid.uuid4(),
                 'successful': successful}

        log_string = ''
        for key, value in kwargs.iteritems():
            if key == 'challenge' or key == 'response':
                entry[key] = repr(value)
            else:
                entry[key] = value
                log_string += '{0}:{1}, '.format(key, value)

        self.login_attempts.append(entry)
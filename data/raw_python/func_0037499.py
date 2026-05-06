def getRegisterUserInfo(self, svctype = "Android NDrive App ver", auth = 0):
        """Retrieve information about useridx

        :param svctype: Information about the platform you are using right now.
        :param auth: Authentication type

        :return: ``True`` when success or ``False`` when failed
        """
        data = {'userid': self.user_id,
                'svctype': svctype,
                'auth': auth
               }

        s, metadata = self.GET('getRegisterUserInfo', data)

        if s is True:
            self.useridx = metadata['useridx']
            return True, metadata
        else:
            return False, metadata
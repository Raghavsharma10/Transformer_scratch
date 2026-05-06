def connect_to_another_user(self, user, password, token=None, is_public=False):
        """
        Authenticates user with the same tenant as current platform using and returns new platform to user.
        :rtype: QubellPlatform
        :param str user: user email
        :param str password: user password
        :param str token: session token
        :param bool is_public: either to use public or private api (public is not fully supported use with caution)
        :return: New Platform instance
        """
        return QubellPlatform.connect(self._router.base_url, user, password, token, is_public)
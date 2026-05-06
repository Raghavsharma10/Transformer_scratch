def connect(tenant=None, user=None, password=None, token=None, is_public=False):
        """
        Authenticates user and returns new platform to user.
        This is an entry point to start working with Qubell Api.
        :rtype: QubellPlatform
        :param str tenant: url to tenant, default taken from 'QUBELL_TENANT'
        :param str user: user email, default taken from 'QUBELL_USER'
        :param str password: user password, default taken from 'QUBELL_PASSWORD'
        :param str token: session token, default taken from 'QUBELL_TOKEN'
        :param bool is_public: either to use public or private api (public is not fully supported use with caution)
        :return: New Platform instance
        """
        if not is_public:
            router = PrivatePath(tenant)
        else:
            router = PublicPath(tenant)
            router.public_api_in_use = is_public

        if token or (user and password):
            router.connect(user, password, token)

        return QubellPlatform().init_router(router)
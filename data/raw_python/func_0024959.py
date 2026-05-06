def create_uaa(self, admin_secret, **kwargs):
        """
        Creates an instance of UAA Service.

        :param admin_secret: The secret password for administering the service
            such as adding clients and users.
        """
        uaa = predix.admin.uaa.UserAccountAuthentication(**kwargs)
        if not uaa.exists():
            uaa.create(admin_secret, **kwargs)

        uaa.add_to_manifest(self)
        return uaa
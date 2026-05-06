def create_client(self, client_id=None, client_secret=None, uaa=None):
        """
        Create a client and add it to the manifest.

        :param client_id: The client id used to authenticate as a client
            in UAA.

        :param client_secret: The secret password used by a client to
            authenticate and generate a UAA token.

        :param uaa: The UAA to create client with
        """
        if not uaa:
            uaa = predix.admin.uaa.UserAccountAuthentication()

        # Client id and secret can be generated if not provided as arguments

        if not client_id:
            client_id = uaa._create_id()

        if not client_secret:
            client_secret = uaa._create_secret()

        uaa.create_client(client_id, client_secret)
        uaa.add_client_to_manifest(client_id, client_secret, self)
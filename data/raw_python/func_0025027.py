def add_client_to_manifest(self, client_id, client_secret, manifest):
        """
        Add the given client / secret to the manifest for use in
        the application.
        """
        client_id_key = 'PREDIX_APP_CLIENT_ID'
        manifest.add_env_var(client_id_key, client_id)

        client_secret_key = 'PREDIX_APP_CLIENT_SECRET'
        manifest.add_env_var(client_secret_key, client_secret)

        manifest.write_manifest()
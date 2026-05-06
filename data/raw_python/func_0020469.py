def set_data_from_reactor_config(self):
        """
        Sets data from reactor config
        """
        reactor_config_override = self.user_params.reactor_config_override.value
        reactor_config_map = self.user_params.reactor_config_map.value
        data = None

        if reactor_config_override:
            data = reactor_config_override
        elif reactor_config_map:
            config_map = self.osbs_api.get_config_map(reactor_config_map)
            data = config_map.get_data_by_key('config.yaml')

        if not data:
            if self.user_params.flatpak.value:
                raise OsbsValidationException("flatpak_base_image must be provided")
            else:
                return

        source_registry_key = 'source_registry'
        registry_organization_key = 'registries_organization'
        req_secrets_key = 'required_secrets'
        token_secrets_key = 'worker_token_secrets'
        flatpak_key = 'flatpak'
        flatpak_base_image_key = 'base_image'

        if source_registry_key in data:
            self.source_registry = data[source_registry_key]
        if registry_organization_key in data:
            self.organization = data[registry_organization_key]

        if self.user_params.flatpak.value:
            flatpack_base_image = data.get(flatpak_key, {}).get(flatpak_base_image_key, None)
            if flatpack_base_image:
                self.base_image = flatpack_base_image
                self.user_params.base_image.value = flatpack_base_image
            else:
                raise OsbsValidationException("flatpak_base_image must be provided")

        required_secrets = data.get(req_secrets_key, [])
        token_secrets = data.get(token_secrets_key, [])
        self._set_required_secrets(required_secrets, token_secrets)
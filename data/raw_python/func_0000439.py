def init_common_services(self, with_cloud_account=True, zone_name=None):
        """
        Initialize common service,
        When 'zone_name' is defined " at $zone_name" is added to service names
        :param bool with_cloud_account:
        :param str zone_name:
        :return: OR tuple(Workflow, Vault), OR tuple(Workflow, Vault, CloudAccount) with services
        """
        zone_names = ZoneConstants(zone_name)
        type_to_app = lambda t: self.organization.applications[system_application_types.get(t, t)]
        wf_service = self.organization.service(name=zone_names.DEFAULT_WORKFLOW_SERVICE,
                                               application=type_to_app(WORKFLOW_SERVICE_TYPE),
                                               environment=self)
        key_service = self.organization.service(name=zone_names.DEFAULT_CREDENTIAL_SERVICE,
                                                application=type_to_app(COBALT_SECURE_STORE_TYPE),
                                                environment=self)
        assert wf_service.running()
        assert key_service.running()
        if not with_cloud_account:
            with self as env:
                env.add_service(wf_service, force=True)
                env.add_service(key_service, force=True)
            return wf_service, key_service

        cloud_account_service = self.organization.instance(name=zone_names.DEFAULT_CLOUD_ACCOUNT_SERVICE,
                                                           application=type_to_app(CLOUD_ACCOUNT_TYPE),
                                                           environment=self,
                                                           parameters=PROVIDER_CONFIG,
                                                           destroyInterval=0)
        # Imidiate adding to env cause CA not to drop destroy interval. Known issue 6132. So, add service as instance with
        # destroyInterval set to 'never'
        assert cloud_account_service.running()

        with self as env:
            env.add_service(wf_service, force=True)
            env.add_service(key_service, force=True)
            env.add_service(cloud_account_service, force=True)
        return wf_service, key_service, cloud_account_service
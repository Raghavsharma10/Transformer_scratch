def default(self):
        """
            Returns environment marked as default.
            When Zone is set marked default makes no sense, special env with proper Zone is returned.
        """
        if ZONE_NAME:
            log.info("Getting or creating default environment for zone with name '{0}'".format(DEFAULT_ENV_NAME()))
            zone_id = self.organization.zones[ZONE_NAME].id
            return self.organization.get_or_create_environment(name=DEFAULT_ENV_NAME(), zone=zone_id)

        def_envs = [env_j["id"] for env_j in self.json() if env_j["isDefault"]]

        if len(def_envs) > 1:
            log.warning('Found more than one default environment. Picking last.')
            return self[def_envs[-1]]
        elif len(def_envs) == 1:
            return self[def_envs[0]]
        raise exceptions.NotFoundError('Unable to get default environment')
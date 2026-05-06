def get_build_request(self, build_type=None, inner_template=None,
                          outer_template=None, customize_conf=None,
                          arrangement_version=DEFAULT_ARRANGEMENT_VERSION):
        """
        return instance of BuildRequest or BuildRequestV2

        :param build_type: str, unused
        :param inner_template: str, name of inner template for BuildRequest
        :param outer_template: str, name of outer template for BuildRequest
        :param customize_conf: str, name of customization config for BuildRequest
        :param arrangement_version: int, value of the arrangement version

        :return: instance of BuildRequest or BuildRequestV2
        """
        if build_type is not None:
            warnings.warn("build types are deprecated, do not use the build_type argument")

        validate_arrangement_version(arrangement_version)

        if not arrangement_version or arrangement_version < REACTOR_CONFIG_ARRANGEMENT_VERSION:
            build_request = BuildRequest(
                build_json_store=self.os_conf.get_build_json_store(),
                inner_template=inner_template,
                outer_template=outer_template,
                customize_conf=customize_conf)
        else:
            build_request = BuildRequestV2(
                build_json_store=self.os_conf.get_build_json_store(),
                outer_template=outer_template,
                customize_conf=customize_conf)

        # Apply configured resource limits.
        cpu_limit = self.build_conf.get_cpu_limit()
        memory_limit = self.build_conf.get_memory_limit()
        storage_limit = self.build_conf.get_storage_limit()
        if (cpu_limit is not None or
                memory_limit is not None or
                storage_limit is not None):
            build_request.set_resource_limits(cpu=cpu_limit,
                                              memory=memory_limit,
                                              storage=storage_limit)

        return build_request
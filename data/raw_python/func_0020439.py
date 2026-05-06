def adjust_for_registry_api_versions(self):
        """
        Enable/disable plugins depending on supported registry API versions
        """
        versions = self.spec.registry_api_versions.value

        if 'v2' not in versions:
            raise OsbsValidationException('v1-only docker registry API is not supported')

        try:
            push_conf = self.dj.dock_json_get_plugin_conf('postbuild_plugins',
                                                          'tag_and_push')
            tag_and_push_registries = push_conf['args']['registries']
        except (KeyError, IndexError):
            tag_and_push_registries = {}

        if 'v1' not in versions:
            # Remove v1-only plugins
            for phase, name in [('postbuild_plugins', 'pulp_push')]:
                logger.info("removing v1-only plugin: %s", name)
                self.dj.remove_plugin(phase, name)

            # remove extra tag_and_push config
            self.remove_tag_and_push_registries(tag_and_push_registries, 'v1')

        # Remove 'version' from tag_and_push plugin config as it's no
        # longer needed
        for regdict in tag_and_push_registries.values():
            if 'version' in regdict:
                del regdict['version']
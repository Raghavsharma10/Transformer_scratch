def render_pulp_tag(self):
        """
        Configure the pulp_tag plugin.
        """
        if not self.dj.dock_json_has_plugin_conf('postbuild_plugins',
                                                 'pulp_tag'):
            return

        pulp_registry = self.spec.pulp_registry.value
        if pulp_registry:
            self.dj.dock_json_set_arg('postbuild_plugins', 'pulp_tag',
                                      'pulp_registry_name', pulp_registry)

            # Verify we have either a secret or username/password
            if self.spec.pulp_secret.value is None:
                conf = self.dj.dock_json_get_plugin_conf('postbuild_plugins',
                                                         'pulp_tag')
                args = conf.get('args', {})
                if 'username' not in args:
                    raise OsbsValidationException("Pulp registry specified "
                                                  "but no auth config")
        else:
            # If no pulp registry is specified, don't run the pulp plugin
            logger.info("removing pulp_tag from request, "
                        "requires pulp_registry")
            self.dj.remove_plugin("postbuild_plugins", "pulp_tag")
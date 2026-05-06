def render_pulp_sync(self):
        """
        If a pulp registry is specified, use the pulp plugin as well as the
        delete_from_registry to delete the image after sync
        """
        if not self.dj.dock_json_has_plugin_conf('postbuild_plugins',
                                                 'pulp_sync'):
            return

        pulp_registry = self.spec.pulp_registry.value

        # Find which registry to use
        docker_registry = None
        registry_secret = None
        registries = zip_longest(self.spec.registry_uris.value,
                                 self.spec.registry_secrets.value)
        for registry, secret in registries:
            if registry.version == 'v2':
                # First specified v2 registry is the one we'll tell pulp
                # to sync from. Keep the http prefix -- pulp wants it.
                docker_registry = registry.uri
                registry_secret = secret
                logger.info("using docker v2 registry %s for pulp_sync",
                            docker_registry)
                break

        if pulp_registry and docker_registry:
            self.dj.dock_json_set_arg('postbuild_plugins', 'pulp_sync',
                                      'pulp_registry_name', pulp_registry)

            self.dj.dock_json_set_arg('postbuild_plugins', 'pulp_sync',
                                      'docker_registry', docker_registry)

            if registry_secret:
                self.set_secret_for_plugin(registry_secret,
                                           plugin=('postbuild_plugins',
                                                   'pulp_sync',
                                                   'registry_secret_path'))

            # Verify we have a pulp secret
            if self.spec.pulp_secret.value is None:
                raise OsbsValidationException("Pulp registry specified "
                                              "but no auth config")

            source_registry = self.spec.source_registry_uri.value
            perform_delete = (source_registry is None or
                              source_registry.docker_uri != registry.docker_uri)
            if perform_delete:
                push_conf = self.dj.dock_json_get_plugin_conf('exit_plugins',
                                                              'delete_from_registry')
                args = push_conf.setdefault('args', {})
                delete_registries = args.setdefault('registries', {})
                placeholder = '{{REGISTRY_URI}}'

                # use passed in params like 'insecure' if available
                if placeholder in delete_registries:
                    regdict = delete_registries[placeholder].copy()
                    del delete_registries[placeholder]
                else:
                    regdict = {}

                if registry_secret:
                    regdict['secret'] = \
                        os.path.join(SECRETS_PATH, registry_secret)
                    # tag_and_push configured the registry secret, no neet to set it again

                delete_registries[docker_registry] = regdict

                self.dj.dock_json_set_arg('exit_plugins', 'delete_from_registry',
                                          'registries', delete_registries)
            else:
                logger.info("removing delete_from_registry from request, "
                            "source and target registry are identical")
                self.dj.remove_plugin("exit_plugins", "delete_from_registry")
        else:
            # If no pulp registry is specified, don't run the pulp plugin
            logger.info("removing pulp_sync+delete_from_registry from request, "
                        "requires pulp_registry and a v2 registry")
            self.dj.remove_plugin("postbuild_plugins", "pulp_sync")
            self.dj.remove_plugin("exit_plugins", "delete_from_registry")
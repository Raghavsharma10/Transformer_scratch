def render_group_manifests(self):
        """
        Configure the group_manifests plugin. Group is always set to false for now.
        """
        if not self.dj.dock_json_has_plugin_conf('postbuild_plugins', 'group_manifests'):
            return

        push_conf = self.dj.dock_json_get_plugin_conf('postbuild_plugins',
                                                      'group_manifests')
        args = push_conf.setdefault('args', {})
        # modify registries in place
        registries = args.setdefault('registries', {})
        placeholder = '{{REGISTRY_URI}}'

        if placeholder in registries:
            for registry, secret in zip_longest(self.spec.registry_uris.value,
                                                self.spec.registry_secrets.value):
                if not registry.uri:
                    continue
                regdict = registries[placeholder].copy()
                regdict['version'] = registry.version
                if secret:
                    regdict['secret'] = os.path.join(SECRETS_PATH, secret)
                registries[registry.docker_uri] = regdict
            del registries[placeholder]

        self.dj.dock_json_set_arg('postbuild_plugins', 'group_manifests',
                                  'group', self.spec.group_manifests.value)
        goarch = {}
        for platform, architecture in self.platform_descriptors.items():
            goarch[platform] = architecture['architecture']
        self.dj.dock_json_set_arg('postbuild_plugins', 'group_manifests',
                                  'goarch', goarch)
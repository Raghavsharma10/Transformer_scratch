def render_fetch_maven_artifacts(self):
        """Configure fetch_maven_artifacts plugin"""
        phase = 'prebuild_plugins'
        plugin = 'fetch_maven_artifacts'
        if not self.dj.dock_json_has_plugin_conf(phase, plugin):
            return

        koji_hub = self.spec.kojihub.value
        koji_root = self.spec.kojiroot.value

        if not koji_hub and not koji_root:
            logger.info('Removing %s because kojihub and kojiroot were not specified', plugin)
            self.dj.remove_plugin(phase, plugin)
            return

        self.dj.dock_json_set_arg(phase, plugin, 'koji_hub', koji_hub)
        self.dj.dock_json_set_arg(phase, plugin, "koji_root", koji_root)

        if self.spec.artifacts_allowed_domains.value:
            self.dj.dock_json_set_arg(phase, plugin, 'allowed_domains',
                                      self.spec.artifacts_allowed_domains.value)
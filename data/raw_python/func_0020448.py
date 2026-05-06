def render_pulp_pull(self):
        """
        If a pulp registry is specified, use pulp_pull plugin
        """
        # pulp_pull is a multi-phase plugin
        phases = ('postbuild_plugins', 'exit_plugins')
        plugin = 'pulp_pull'
        for phase in phases:
            if not self.dj.dock_json_has_plugin_conf(phase, plugin):
                continue

            pulp_registry = self.spec.pulp_registry.value
            if not pulp_registry:
                logger.info("removing %s from request, requires pulp_registry", pulp_registry)
                self.dj.remove_plugin(phase, plugin)
                continue

            if not self.spec.kojihub.value:
                logger.info('Removing %s because no kojihub was specified', plugin)
                self.dj.remove_plugin(phase, plugin)
                continue

            if self.spec.prefer_schema1_digest.value is not None:
                self.dj.dock_json_set_arg(phase, 'pulp_pull',
                                          'expect_v2schema2',
                                          not self.spec.prefer_schema1_digest.value)
def render_bump_release(self):
        """
        If the bump_release plugin is present, configure it
        """
        phase = 'prebuild_plugins'
        plugin = 'bump_release'
        if not self.dj.dock_json_has_plugin_conf(phase, plugin):
            return

        if self.spec.release.value:
            logger.info('removing %s from request as release already specified',
                        plugin)
            self.dj.remove_plugin(phase, plugin)
            return

        hub = self.spec.kojihub.value
        if not hub:
            logger.info('removing %s from request as koji hub not specified',
                        plugin)
            self.dj.remove_plugin(phase, plugin)
            return

        self.dj.dock_json_set_arg(phase, plugin, 'hub', hub)

        # For flatpak, we want a name-version-release of
        # <name>-<stream>-<module_build_version>.<n>, where the .<n> makes
        # sure that the build is unique in Koji
        if self.spec.flatpak.value:
            self.dj.dock_json_set_arg(phase, plugin, 'append', True)
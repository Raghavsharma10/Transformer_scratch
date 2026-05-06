def render_bump_release(self):
        """
        If the bump_release plugin is present, configure it
        """
        phase = 'prebuild_plugins'
        plugin = 'bump_release'
        if not self.pt.has_plugin_conf(phase, plugin):
            return

        if self.user_params.release.value:
            self.pt.remove_plugin(phase, plugin, 'release value supplied as user parameter')
            return

        # For flatpak, we want a name-version-release of
        # <name>-<stream>-<module_build_version>.<n>, where the .<n> makes
        # sure that the build is unique in Koji
        if self.user_params.flatpak.value:
            self.pt.set_plugin_arg(phase, plugin, 'append', True)
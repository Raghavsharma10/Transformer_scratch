def render_koji(self):
        """
        if there is yum repo in user params, don't pick stuff from koji
        """
        phase = 'prebuild_plugins'
        plugin = 'koji'
        if not self.pt.has_plugin_conf(phase, plugin):
            return

        if self.user_params.yum_repourls.value:
            self.pt.remove_plugin(phase, plugin, 'there is a yum repo user parameter')
        elif not self.pt.set_plugin_arg_valid(phase, plugin, "target",
                                              self.user_params.koji_target.value):
            self.pt.remove_plugin(phase, plugin, 'no koji target supplied in user parameters')
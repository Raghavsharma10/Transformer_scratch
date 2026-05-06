def render_koji(self):
        """
        if there is yum repo specified, don't pick stuff from koji
        """
        phase = 'prebuild_plugins'
        plugin = 'koji'
        if not self.dj.dock_json_has_plugin_conf(phase, plugin):
            return

        if self.spec.yum_repourls.value:
            logger.info("removing koji from request "
                        "because there is yum repo specified")
            self.dj.remove_plugin(phase, plugin)
        elif not (self.spec.koji_target.value and
                  self.spec.kojiroot.value and
                  self.spec.kojihub.value):
            logger.info("removing koji from request as not specified")
            self.dj.remove_plugin(phase, plugin)
        else:
            self.dj.dock_json_set_arg(phase, plugin,
                                      "target", self.spec.koji_target.value)
            self.dj.dock_json_set_arg(phase, plugin,
                                      "root", self.spec.kojiroot.value)
            self.dj.dock_json_set_arg(phase, plugin,
                                      "hub", self.spec.kojihub.value)
            if self.spec.proxy.value:
                self.dj.dock_json_set_arg(phase, plugin,
                                          "proxy", self.spec.proxy.value)
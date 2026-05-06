def render_sendmail(self):
        """
        if we have smtp_host and smtp_from, configure sendmail plugin,
        else remove it
        """
        phase = 'exit_plugins'
        plugin = 'sendmail'
        if not self.dj.dock_json_has_plugin_conf(phase, plugin):
            return

        if self.spec.smtp_host.value and self.spec.smtp_from.value:
            self.dj.dock_json_set_arg(phase, plugin, 'url',
                                      self.spec.builder_openshift_url.value)
            self.dj.dock_json_set_arg(phase, plugin, 'smtp_host',
                                      self.spec.smtp_host.value)
            self.dj.dock_json_set_arg(phase, plugin, 'from_address',
                                      self.spec.smtp_from.value)
        else:
            logger.info("removing sendmail from request, "
                        "requires smtp_host and smtp_from")
            self.dj.remove_plugin(phase, plugin)
            return

        if self.spec.kojihub.value and self.spec.kojiroot.value:
            self.dj.dock_json_set_arg(phase, plugin,
                                      'koji_hub', self.spec.kojihub.value)
            self.dj.dock_json_set_arg(phase, plugin,
                                      "koji_root", self.spec.kojiroot.value)

            if self.spec.smtp_to_submitter.value:
                self.dj.dock_json_set_arg(phase, plugin, 'to_koji_submitter',
                                          self.spec.smtp_to_submitter.value)
            if self.spec.smtp_to_pkgowner.value:
                self.dj.dock_json_set_arg(phase, plugin, 'to_koji_pkgowner',
                                          self.spec.smtp_to_pkgowner.value)

        if self.spec.smtp_additional_addresses.value:
            self.dj.dock_json_set_arg(phase, plugin, 'additional_addresses',
                                      self.spec.smtp_additional_addresses.value)

        if self.spec.smtp_error_addresses.value:
            self.dj.dock_json_set_arg(phase, plugin,
                                      'error_addresses', self.spec.smtp_error_addresses.value)

        if self.spec.smtp_email_domain.value:
            self.dj.dock_json_set_arg(phase, plugin,
                                      'email_domain', self.spec.smtp_email_domain.value)
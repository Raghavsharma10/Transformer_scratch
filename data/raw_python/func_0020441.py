def adjust_for_scratch(self):
        """
        Remove certain plugins in order to handle the "scratch build"
        scenario. Scratch builds must not affect subsequent builds,
        and should not be imported into Koji.
        """
        if self.scratch:
            self.template['spec'].pop('triggers', None)

            remove_plugins = [
                ("prebuild_plugins", "koji_parent"),
                ("postbuild_plugins", "compress"),  # required only to make an archive for Koji
                ("postbuild_plugins", "pulp_pull"),  # required only to make an archive for Koji
                ("postbuild_plugins", "koji_upload"),
                ("postbuild_plugins", "fetch_worker_metadata"),
                ("postbuild_plugins", "compare_components"),
                ("postbuild_plugins", "import_image"),
                ("exit_plugins", "koji_promote"),
                ("exit_plugins", "koji_import"),
                ("exit_plugins", "koji_tag_build"),
                ("exit_plugins", "remove_worker_metadata"),
                ("exit_plugins", "import_image"),
            ]

            if not self.has_tag_suffixes_placeholder():
                remove_plugins.append(("postbuild_plugins", "tag_from_config"))

            for when, which in remove_plugins:
                logger.info("removing %s from scratch build request",
                            which)
                self.dj.remove_plugin(when, which)

            if self.dj.dock_json_has_plugin_conf('postbuild_plugins',
                                                 'tag_by_labels'):
                self.dj.dock_json_set_arg('postbuild_plugins', 'tag_by_labels',
                                          'unique_tag_only', True)

            self.set_label('scratch', 'true')
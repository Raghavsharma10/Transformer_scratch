def adjust_for_scratch(self):
        """
        Remove certain plugins in order to handle the "scratch build"
        scenario. Scratch builds must not affect subsequent builds,
        and should not be imported into Koji.
        """
        if self.user_params.scratch.value:
            remove_plugins = [
                ("prebuild_plugins", "koji_parent"),
                ("postbuild_plugins", "compress"),  # required only to make an archive for Koji
                ("postbuild_plugins", "pulp_pull"),  # required only to make an archive for Koji
                ("postbuild_plugins", "compare_components"),
                ("postbuild_plugins", "import_image"),
                ("exit_plugins", "koji_promote"),
                ("exit_plugins", "koji_tag_build"),
                ("exit_plugins", "import_image"),
                ("prebuild_plugins", "check_and_set_rebuild"),
                ("prebuild_plugins", "stop_autorebuild_if_disabled")
            ]

            if not self.has_tag_suffixes_placeholder():
                remove_plugins.append(("postbuild_plugins", "tag_from_config"))

            for when, which in remove_plugins:
                self.pt.remove_plugin(when, which, 'removed from scratch build request')
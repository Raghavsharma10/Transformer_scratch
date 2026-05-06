def adjust_for_flatpak(self):
        """
        Remove plugins that don't work when building Flatpaks
        """
        if self.user_params.flatpak.value:
            remove_plugins = [
                ("prebuild_plugins", "resolve_composes"),
                # We'll extract the filesystem anyways for a Flatpak instead of exporting
                # the docker image directly, so squash just slows things down.
                ("prepublish_plugins", "squash"),
                # Pulp can't currently handle Flatpaks, which are OCI images
                ("postbuild_plugins", "pulp_push"),
                ("postbuild_plugins", "pulp_tag"),
                ("postbuild_plugins", "pulp_sync"),
                ("exit_plugins", "pulp_publish"),
                ("exit_plugins", "pulp_pull"),
                # delete_from_registry is used for deleting builds from the temporary registry
                # that pulp_sync mirrors from.
                ("exit_plugins", "delete_from_registry"),
            ]
            for when, which in remove_plugins:
                self.pt.remove_plugin(when, which, 'not needed for flatpak build')
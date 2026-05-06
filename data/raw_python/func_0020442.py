def adjust_for_custom_base_image(self):
        """
        Disable plugins to handle builds depending on whether
        or not this is a build from a custom base image.
        """
        plugins = []
        if self.is_custom_base_image():
            # Plugins irrelevant to building base images.
            plugins.append(("prebuild_plugins", "pull_base_image"))
            plugins.append(("prebuild_plugins", "koji_parent"))
            plugins.append(("prebuild_plugins", "inject_parent_image"))
            msg = "removing %s from custom image build request"
        else:
            # Plugins not needed for building non base images.
            plugins.append(("prebuild_plugins", "add_filesystem"))
            msg = "removing %s from non custom image build request"

        for when, which in plugins:
            logger.info(msg, which)
            self.dj.remove_plugin(when, which)
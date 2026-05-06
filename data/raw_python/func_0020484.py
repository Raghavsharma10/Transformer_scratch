def render_pull_base_image(self):
        """Configure pull_base_image"""
        phase = 'prebuild_plugins'
        plugin = 'pull_base_image'

        if self.user_params.parent_images_digests.value:
            self.pt.set_plugin_arg(phase, plugin, 'parent_images_digests',
                                   self.user_params.parent_images_digests.value)
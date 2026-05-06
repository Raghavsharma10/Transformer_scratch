def render_import_image(self, use_auth=None):
        """
        Configure the import_image plugin
        """
        # import_image is a multi-phase plugin
        if self.user_params.imagestream_name.value is None:
            self.pt.remove_plugin('exit_plugins', 'import_image',
                                  'imagestream not in user parameters')
        elif self.pt.has_plugin_conf('exit_plugins', 'import_image'):
            self.pt.set_plugin_arg('exit_plugins', 'import_image', 'imagestream',
                                   self.user_params.imagestream_name.value)
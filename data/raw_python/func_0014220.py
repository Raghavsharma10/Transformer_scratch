def build_default_filepath(self):
        '''Called when 'filepath' is not defined in the settings'''
        return os.path.join(
            self.app_config.name,
            'scripts',
            self.template_relpath + '.js',
        )
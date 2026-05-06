def override_default_templates(self):
        """
        Override the default emails already defined by other apps
        """
        if plugs_mail_settings['OVERRIDE_TEMPLATE_DIR']:
            dir_ = plugs_mail_settings['OVERRIDE_TEMPLATE_DIR']
            for file_ in os.listdir(dir_):
                if file_.endswith(('.html', 'txt')):
                    self.overrides[file_] = dir_
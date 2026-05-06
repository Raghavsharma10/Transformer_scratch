def get_add_plugin_kwargs(self, page, no, placeholder, language_code,
                              lang_name):
        """
        Return "content" for create the plugin.
        Called from self.add_plugins()
        """
        return {
            "plugin_type":
            'TextPlugin',  # djangocms_text_ckeditor
            "body":
            self.get_dummy_text(page, no, placeholder, language_code,
                                lang_name)
        }
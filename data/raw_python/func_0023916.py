def get_template_names(self):
        '''
        Build the list of templates related to this user
        '''

        # Get user template
        template_model = getattr(self, 'template_model', "{0}/{1}_{2}".format(self._appname.lower(), self._modelname.lower(), self.get_template_names_key))
        template_model_ext = getattr(self, 'template_model_ext', 'html')
        templates = get_template(template_model, self.user, self.language, template_model_ext, raise_error=False)
        if type(templates) == list:
            templates.append("codenerix/{0}.html".format(self.get_template_names_key))

        # Return thet of templates
        return templates
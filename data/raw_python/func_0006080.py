def execute_template(self, template_name, variables, args=None):
        """
        Execute script from a template

        @type template_name:    str
        @value template_name:   Script template to implement
        @type args:             dict
        @value args:            Dictionary representing command line args

        @rtype:                 bool
        @rtype:                 Success or failure
        """
        js_text = self.build_js_from_template(template_name, variables)
        try:
            self.execute_script(js_text, args)
        except WebDriverException:
            return False
        return True
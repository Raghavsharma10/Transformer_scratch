def execute_template_and_return_result(self, template_name, variables, args=None):
        """
        Execute script from a template and return result

        @type template_name:    str
        @value template_name:   Script template to implement
        @type variables:        dict
        @value variables:       Dictionary representing template construction args
        @type args:             dict
        @value args:            Dictionary representing command line args

        @rtype:                 int
        @rtype:                 exit code
        """
        js_text = self.build_js_from_template(template_name, variables)
        return self.execute_script(js_text, args)
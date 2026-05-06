def get_output(self):
        """
        Returns the generated JavaScript code.

        Returns:
            str
        """
        # generate the JS function string
        template_function = TEMPLATE_WRAPPER.format(
            function_name=self.js_function_name,
            template_code=self.output.getvalue()
        ).strip()

        # get the correct module format template
        module_format = JS_MODULE_FORMATS[self.js_module_format]

        # generate the module code
        return module_format(self.dependencies, template_function)
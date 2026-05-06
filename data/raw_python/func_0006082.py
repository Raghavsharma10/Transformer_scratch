def build_js_from_template(self, template_file, variables):
        """
        Build a JS script from a template and args

        @type template_file:    str
        @param template_file:   Script template to implement; can be the name of a built-in script or full filepath to
                                a js file that contains the script. E.g. 'clickElementTemplate.js',
                                'clickElementTemplate', and '/path/to/custom/template/script.js' are all acceptable
        @type variables:        dict
        @param variables:       Dictionary representing template construction args

        @rtype:                 int
        @rtype:                 exit code
        """
        template_variable_character = '%'

        # raise an exception if user passed non-dictionary variables
        if not isinstance(variables, dict):
            raise TypeError('You must use a dictionary to populate variables in a javascript template')

        # This filename is not a full file, attempt to locate the file in built-in templates
        if not os.path.isfile(template_file):
            # append the .js extension if not included
            if '.js' not in template_file:
                template_file += '.js'

            # find the template and read the text into a string variable
            templates_dir = os.path.join(os.path.dirname(__file__), 'jsTemplates')
            template_full_path = os.path.join(templates_dir, template_file)
        # The filename specified should be the full path
        else:
            template_full_path = template_file

        # Ensure that the file exists
        if not os.path.isfile(template_full_path):
            raise ValueError('File "{}" was not found; you must specify the name of a built-in javascript template '
                             'or the full filepath of a custom template'.format(template_full_path))

        try:
            js_text = open(template_full_path).read()
        except IOError:
            raise IOError('The template was not found or did not have read permissions: {}'.format(template_full_path))

        # replace all variables that match the keys in 'variables' dict
        for key in variables.keys():
            # double escape single and double quotes after variable replacement
            if hasattr(variables[key], 'replace'):
                variables[key] = variables[key].replace("'", "\\'")
                variables[key] = variables[key].replace('"', '\\"')
            else: # variable is not a string
                variables[key] = str(variables[key])

            js_text = js_text.replace(template_variable_character + key, variables[key])

        return js_text
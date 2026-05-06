def from_string(self, template_code):
        '''
        Compiles a template from the given string.
        This is one of the required methods of Django template engines.
        '''
        dmp = apps.get_app_config('django_mako_plus')
        mako_template = Template(template_code, imports=dmp.template_imports, input_encoding=dmp.options['DEFAULT_TEMPLATE_ENCODING'])
        return MakoTemplateAdapter(mako_template)
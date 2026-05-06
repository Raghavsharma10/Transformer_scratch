def get_template(self, template_name):
        '''
        Retrieves a template object from the pattern "app_name/template.html".
        This is one of the required methods of Django template engines.

        Because DMP templates are always app-specific (Django only searches
        a global set of directories), the template_name MUST be in the format:
        "app_name/template.html" (even on Windows).  DMP splits the template_name
        string on the slash to get the app name and template name.

        Template rendering can be limited to a specific def/block within the template
        by specifying `#def_name`, e.g. `myapp/mytemplate.html#myblockname`.
        '''
        dmp = apps.get_app_config('django_mako_plus')
        match = RE_TEMPLATE_NAME.match(template_name)
        if match is None or match.group(1) is None or match.group(3) is None:
            raise TemplateDoesNotExist('Invalid template_name format for a DMP template.  This method requires that the template name be in app_name/template.html format (separated by slash).')
        if not dmp.is_registered_app(match.group(1)):
            raise TemplateDoesNotExist('Not a DMP app, so deferring to other template engines for this template')
        return self.get_template_loader(match.group(1)).get_template(match.group(3), def_name=match.group(5))
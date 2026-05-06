def get_template(self, template, def_name=None):
        '''Retrieve a *Django* API template object for the given template name, using the app_path and template_subdir
           settings in this object.  This method still uses the corresponding Mako template and engine, but it
           gives a Django API wrapper around it so you can use it the same as any Django template.

           If def_name is provided, template rendering will be limited to the named def/block (see Mako docs).

           This method corresponds to the Django templating system API.
           A Django exception is raised if the template is not found or cannot compile.
        '''
        try:
            # wrap the mako template in an adapter that gives the Django template API
            return MakoTemplateAdapter(self.get_mako_template(template), def_name)

        except (TopLevelLookupException, TemplateLookupException) as e: # Mako exception raised
            tdne = TemplateDoesNotExist('Template "%s" not found in search path: %s.' % (template, self.template_search_dirs))
            if settings.DEBUG:
                tdne.template_debug = get_template_debug(template, e)
            raise tdne from e

        except (CompileException, SyntaxException) as e: # Mako exception raised
            tse = TemplateSyntaxError('Template "%s" raised an error: %s' % (template, e))
            if settings.DEBUG:
                tse.template_debug = get_template_debug(template, e)
            raise tse from e
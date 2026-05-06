def get_mako_template(self, template, force=False):
        '''Retrieve the real *Mako* template object for the given template name without any wrapper,
           using the app_path and template_subdir settings in this object.

           This method is an alternative to get_template().  Use it when you need the actual Mako template object.
           This method raises a Mako exception if the template is not found or cannot compile.

           If force is True, an empty Mako template will be created when the file does not exist.
           This option is used by the providers part of DMP and normally be left False.
        '''
        if template is None:
            raise TemplateLookupException('Template "%s" not found in search path: %s.' % (template, self.template_search_dirs))
        # get the template
        try:
            template_obj = self.tlookup.get_template(template)
        except TemplateLookupException:
            if not force:
                raise
            template_obj = Template('', filename=os.path.join(self.template_dir, template))

        # get the template
        return template_obj
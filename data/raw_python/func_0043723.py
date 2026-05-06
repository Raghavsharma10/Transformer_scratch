def join_template_name(self, template_name, extension):
        """
        Appends an extension to a template_name or list of template_names.
        """
        if template_name is None:
            return None
        if isinstance(template_name, (list, tuple)):
            return tuple('.'.join([n, extension]) for n in template_name)
        if isinstance(template_name, str_types):
            return '.'.join([template_name, extension])
        raise AssertionError('template_name not of correct type: %r' % type(template_name))
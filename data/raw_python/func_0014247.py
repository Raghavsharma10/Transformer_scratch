def render_to_string(self, template, context=None, def_name=None, subdir='templates'):
        '''App-specific render function that renders templates in the *current app*, attached to the request for convenience'''
        template_adapter = self.get_template_loader(subdir).get_template(template)
        return getattr(template_adapter, 'render')(context=context, request=self.request, def_name=def_name)
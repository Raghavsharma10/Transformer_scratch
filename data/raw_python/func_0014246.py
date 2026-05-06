def render(self, template, context=None, def_name=None, subdir='templates', content_type=None, status=None, charset=None):
        '''App-specific render function that renders templates in the *current app*, attached to the request for convenience'''
        template_adapter = self.get_template_loader(subdir).get_template(template)
        return getattr(template_adapter, 'render_to_response')(context=context, request=self.request, def_name=def_name, content_type=content_type, status=status, charset=charset)
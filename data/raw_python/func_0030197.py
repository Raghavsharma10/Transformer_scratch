def render(self, template_name, __data=None, **kw):
        '''Given a template name and template data.
        Renders a template and returns as string'''
        return self.template.render(template_name,
                                    **self._vars(__data, **kw))
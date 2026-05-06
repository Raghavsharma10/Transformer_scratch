def render_to_response(self, template_name, __data,
                           content_type="text/html"):
        '''Given a template name and template data.
        Renders a template and returns `webob.Response` object'''
        resp = self.render(template_name, __data)
        return Response(resp,
                        content_type=content_type)
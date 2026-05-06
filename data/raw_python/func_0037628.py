def render(self, form=None, **kwargs):
        """Returns the ``HttpResponse`` with the context data"""
        context = self.get_context(**kwargs)
        return self.render_to_response(context)
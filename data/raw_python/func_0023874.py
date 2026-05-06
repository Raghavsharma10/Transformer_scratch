def render_to_response(self, *args, **kwargs):
        '''Canonicalize the URL if the slug changed'''
        if self.request.path != self.object.get_absolute_url():
            return HttpResponseRedirect(self.object.get_absolute_url())
        return super(TalkView, self).render_to_response(*args, **kwargs)
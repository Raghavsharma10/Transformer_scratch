def set_override_rendered(self):
        """ Set self.request.override_renderer if needed. """
        if '' in self.request.accept:
            self.request.override_renderer = self._default_renderer
        elif 'application/json' in self.request.accept:
            self.request.override_renderer = 'nefertari_json'
        elif 'text/plain' in self.request.accept:
            self.request.override_renderer = 'string'
def setup_default_wrappers(self):
        """ Setup defaulf wrappers.

        Wrappers are applied when view method does not return instance
        of Response. In this case nefertari renderers call wrappers and
        handle response generation.
        """
        # Index
        self._after_calls['index'] = [
            wrappers.wrap_in_dict(self.request),
            wrappers.add_meta(self.request),
            wrappers.add_object_url(self.request),
        ]

        # Show
        self._after_calls['show'] = [
            wrappers.wrap_in_dict(self.request),
            wrappers.add_meta(self.request),
            wrappers.add_object_url(self.request),
        ]

        # Create
        self._after_calls['create'] = [
            wrappers.wrap_in_dict(self.request),
            wrappers.add_meta(self.request),
            wrappers.add_object_url(self.request),
        ]

        # Update
        self._after_calls['update'] = [
            wrappers.wrap_in_dict(self.request),
            wrappers.add_meta(self.request),
            wrappers.add_object_url(self.request),
        ]

        # Replace
        self._after_calls['replace'] = [
            wrappers.wrap_in_dict(self.request),
            wrappers.add_meta(self.request),
            wrappers.add_object_url(self.request),
        ]

        # Privacy wrappers
        if self._auth_enabled:
            for meth in ('index', 'show', 'create', 'update', 'replace'):
                self._after_calls[meth] += [
                    wrappers.apply_privacy(self.request),
                ]
            for meth in ('update', 'replace', 'update_many'):
                self._before_calls[meth] += [
                    wrappers.apply_request_privacy(
                        self.Model, self._json_params),
                ]
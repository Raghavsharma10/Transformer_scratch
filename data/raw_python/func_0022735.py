def _process_backend_kwargs(self, kwargs):
        """ Simple utility to retrieve kwargs in predetermined order.
        Also checks whether the values of the backend arguments do not
        violate the backend capabilities.
        """
        # Verify given argument with capability of the backend
        app = self._vispy_canvas.app
        capability = app.backend_module.capability
        if kwargs['context'].shared.name:  # name already assigned: shared
            if not capability['context']:
                raise RuntimeError('Cannot share context with this backend')
        for key in [key for (key, val) in capability.items() if not val]:
            if key in ['context', 'multi_window', 'scroll']:
                continue
            invert = key in ['resizable', 'decorate']
            if bool(kwargs[key]) - invert:
                raise RuntimeError('Config %s is not supported by backend %s'
                                   % (key, app.backend_name))

        # Return items in sequence
        out = SimpleBunch()
        keys = ['title', 'size', 'position', 'show', 'vsync', 'resizable',
                'decorate', 'fullscreen', 'parent', 'context', 'always_on_top',
                ]
        for key in keys:
            out[key] = kwargs[key]
        return out
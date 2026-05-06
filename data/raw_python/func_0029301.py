def _render_response(self, value, system):
        """ Handle response rendering.

        Calls mixin methods according to request.action value.
        """
        super_call = super(DefaultResponseRendererMixin, self)._render_response
        try:
            method_name = 'render_{}'.format(system['request'].action)
        except (KeyError, AttributeError):
            return super_call(value, system)
        method = getattr(self, method_name, None)
        if method is not None:
            common_kw = self._get_common_kwargs(system)
            response = method(value, system, common_kw)
            system['request'].response = response
            return
        return super_call(value, system)
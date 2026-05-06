def _get_graph(self, ctx, bundle, extensions, caller=None):
        """ Run a graph and render the tag contents for each output """
        request = ctx.get('request')
        if request is None:
            request = get_current_request()
        if ':' in bundle:
            config_name, bundle = bundle.split(':')
        else:
            config_name = 'DEFAULT'
        webpack = request.webpack(config_name)
        assets = (caller(a) for a in webpack.get_bundle(bundle, extensions))
        return ''.join(assets)
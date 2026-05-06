def _filter_attrs(self, feature, request):
        """ Remove some attributes from the feature and set the geometry to None
            in the feature based ``attrs`` and the ``no_geom`` parameters. """
        if 'attrs' in request.params:
            attrs = request.params['attrs'].split(',')
            props = feature.properties
            new_props = {}
            for name in attrs:
                if name in props:
                    new_props[name] = props[name]
            feature.properties = new_props
        if asbool(request.params.get('no_geom', False)):
            feature.geometry = None
        return feature
def render(self, data, accepted_media_type=None, renderer_context=None):
        """
        Renders data to HTML, using Django's standard template rendering.

        The template name is determined by (in order of preference):

        1. An explicit .template_name set on the response.
        2. An explicit .template_name set on this class.
        3. The return result of calling view.get_template_names().
        """
        renderer_context = renderer_context or {}
        view = renderer_context['view']
        request = renderer_context['request']
        response = renderer_context['response']
        extra_context = renderer_context['extra_context']
        obj = view.get_object()

        if response.exception:
            template = self.get_exception_template(response)
        else:
            template_names = self.get_template_names(response, view)
            template = self.resolve_template(template_names)

        context = self.resolve_context(data, request, response)

        if response.exception:
            # If there is an exception don't bother calculating data or geometries
            context.update({'template_extra_context': extra_context})
            return template.render(context)

        new_data = {
            "type": "FeatureCollection",
        }

        features = []

        self.setup_icons_dict()

        try:
            popup_template = Template(getattr(obj, 'popup_template', None))
        except TemplateSyntaxError as exception:
            popup_template = Template(exception)

        try:
            marker_template = Template(getattr(obj, 'marker_template', None))
        except TemplateSyntaxError as exception:
            marker_template = Template(exception)

        if isinstance(data, dict):
            features.append(self.process_feature(data, popup_template, marker_template))
        else:
            for feature in data:
                features.append(self.process_feature(feature, popup_template, marker_template))

        new_data['features'] = features
        if 'latitude' and 'longitude' not in extra_context:
            # User didn't specified which latitude and longitude to move the map,
            # determine where to move the map ourselves
            try:
                extra_context['extents'] = self.determine_extents(features)
            except (StopIteration, BoundsError):
                pass

        ret = json.dumps(new_data, cls=self.encoder_class, ensure_ascii=True)

        # On python 2.x json.dumps() returns bytestrings if ensure_ascii=True,
        # but if ensure_ascii=False, the return type is underspecified,
        # and may (or may not) be unicode.
        # On python 3.x json.dumps() returns unicode strings.
        if isinstance(ret, six.text_type):
            ret = bytes(ret.encode(self.charset))

        context.update({'data': ret, 'markers': obj.markers, 'header': obj.template_header})

        if 'geometry' in extra_context:
            extra_context['geometry'] = json.dumps(extra_context['geometry'].__geo_interface__)

        context.update({'template_extra_context': extra_context})

        return template.render(context)
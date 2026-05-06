def get_renderers(self, request, context=None, template_name=None,
                      accept_header=None, formats=None, default_format=None, fallback_formats=None,
                      early=False):
        """
        Returns a list of renderer functions in the order they should be tried.
        
        Tries the format override parameter first, then the Accept header. If
        neither is present, attempt to fall back to self._default_format. If
        a fallback format has been specified, we try that last.
        
        If early is true, don't test renderers to see whether they can handle
        a serialization. This is useful if we're trying to find all relevant
        serializers before we've built a context which they will accept. 
        """
        if formats:
            renderers, seen_formats = [], set()
            for format in formats:
                if format in self.renderers_by_format and format not in seen_formats:
                    renderers.extend(self.renderers_by_format[format])
                    seen_formats.add(format)
        elif accept_header:
            accepts = MediaType.parse_accept_header(accept_header)
            renderers = MediaType.resolve(accepts, self.renderers)
        elif default_format:
            renderers = self.renderers_by_format[default_format]
        else:
            renderers = []

        fallback_formats = fallback_formats if isinstance(fallback_formats, (list, tuple)) else (fallback_formats,)
        for format in fallback_formats:
            for renderer in self.renderers_by_format[format]:
                if renderer not in renderers:
                    renderers.append(renderer)

        if not early and context is not None and template_name:
            renderers = [r for r in renderers if r.test(request, context, template_name)]

        return renderers
def resolve(self, path):
        '''
        Different from Django, this method matches by /app/page/ convention
        using its pattern.  The pattern should create keyword arguments for
        dmp_app, dmp_page.
        '''
        match = super().resolve(path)
        if match:
            try:
                routing_data = RoutingData(
                    match.kwargs.pop('dmp_app', None) or self.dmp.options['DEFAULT_APP'],
                    match.kwargs.pop('dmp_page', None) or self.dmp.options['DEFAULT_PAGE'],
                    match.kwargs.pop('dmp_function', None) or 'process_request',
                    match.kwargs.pop('dmp_urlparams', '').strip(),
                )
                if VERSION < (2, 2):
                    return ResolverMatch(
                        RequestViewWrapper(routing_data),
                        match.args,
                        match.kwargs,
                        url_name=match.url_name,
                        app_names=routing_data.app,
                    )
                else:
                    return ResolverMatch(
                        RequestViewWrapper(routing_data),
                        match.args,
                        match.kwargs,
                        url_name=match.url_name,
                        app_names=routing_data.app,
                        route=str(self.pattern),
                    )
            except ViewDoesNotExist as vdne:
                # we had a pattern match, but we couldn't get a callable using kwargs from the pattern
                # create a "pattern" so the programmer can see what happened
                # this is a hack, but the resolver error page doesn't give other options.
                # the sad face is to catch the dev's attention in Django's printout
                msg = "◉︵◉ Pattern matched, but discovery failed: {}".format(vdne)
                log.debug("%s %s", match.url_name, msg)
                raise Resolver404({
                    # this is a bit convoluted, but it makes the PatternStub work with Django 1.x and 2.x
                    'tried': [[ PatternStub(match.url_name, msg, PatternStub(match.url_name, msg, None)) ]],
                    'path': path,
                })
        raise Resolver404({'path': path})
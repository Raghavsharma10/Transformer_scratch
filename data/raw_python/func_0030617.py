def as_url(self):
        '''
        Reverse object converted to `web.URL`.

        If Reverse is bound to env:
            * try to build relative URL,
            * use current domain name, port and scheme as default
        '''
        if '' in self._scope:
            return self._finalize().as_url

        if not self._is_endpoint:
            raise UrlBuildingError('Not an endpoint {}'.format(repr(self)))

        if self._ready:
            path, host = self._path, self._host
        else:
            return self().as_url

        # XXX there is a little mess with `domain` and `host` terms
        if ':' in host:
            domain, port = host.split(':')
        else:
            domain = host
            port = None

        if self._bound_env:
            request = self._bound_env.request
            scheme_port = {'http': '80',
                           'https': '443'}.get(request.scheme, '80')

            # Domain to compare with the result of build.
            # If both values are equal, domain part can be hidden from result.
            # Take it from route_state, not from env.request, because
            # route_state contains domain values with aliased replaced by their
            # primary value
            primary_domain = self._bound_env._route_state.primary_domain
            host_split = request.host.split(':')
            request_domain = host_split[0]
            request_port = host_split[1] if len(host_split) > 1 else scheme_port
            port = port or request_port

            return URL(path, host=domain or request_domain,
                       port=port if port != scheme_port else None,
                       scheme=request.scheme, fragment=self._fragment,
                       show_host=host and (domain != primary_domain \
                                           or port != request_port))
        return URL(path, host=domain, port=port,
                   fragment=self._fragment, show_host=True)
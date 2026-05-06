def fromRequest(cls, store, request):
        """
        Return a L{LoginPage} which will present the user with a login prompt.

        @type store: L{Store}
        @param store: A I{site} store.

        @type request: L{nevow.inevow.IRequest}
        @param request: The HTTP request which encountered a need for
            authentication.  This will be effectively re-issued after login
            succeeds.

        @return: A L{LoginPage} and the remaining segments to be processed.
        """
        location = URL.fromRequest(request)
        segments = location.pathList(unquote=True, copy=False)
        segments.append(request.postpath[0])
        return cls(store, segments, request.args)
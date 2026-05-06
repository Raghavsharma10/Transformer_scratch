def bindRoute(self, receiver, routeName=_unspecified):
        """
        Create a new route to associate the given route name with the given
        receiver.

        @type routeName: C{unicode} or L{NoneType}
        @param routeName: The identifier for the newly created route.  If
            C{None}, boxes with no route in them will be delivered to this
            receiver.

        @rtype: L{Route}
        """
        if routeName is _unspecified:
            routeName = self.createRouteIdentifier()
        # self._sender may yet be None; if so, this route goes into _unstarted
        # and will have its sender set correctly in startReceivingBoxes below.
        route = Route(self, receiver, routeName)
        mapping = self._routes
        if mapping is None:
            mapping = self._unstarted
        mapping[routeName] = route
        return route
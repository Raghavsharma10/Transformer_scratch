def sendBox(self, box):
        """
        Add the route and send the box.
        """
        if self.remoteRouteName is _unspecified:
            raise RouteNotConnected()
        if self.remoteRouteName is not None:
            box[_ROUTE] = self.remoteRouteName.encode('ascii')
        self.router._sender.sendBox(box)
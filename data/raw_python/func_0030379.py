def ampBoxReceived(self, box):
        """
        Dispatch the given box to the L{IBoxReceiver} associated with the route
        indicated by the box, or handle it directly if there is no route.
        """
        route = box.pop(_ROUTE, None)
        self._routes[route].receiver.ampBoxReceived(box)
def get_router(self):
        """Returns the router for the cluster.  If the cluster reconfigures
        the router will be recreated.  Usually you do not need to interface
        with the router yourself as the cluster's routing client does that
        automatically.

        This returns an instance of :class:`BaseRouter`.
        """
        cached_router = self._router
        ref_age = self._hosts_age

        if cached_router is not None:
            router, router_age = cached_router
            if router_age == ref_age:
                return router

        with self._lock:
            router = self.router_cls(self, **(self.router_options or {}))
            self._router = (router, ref_age)
            return router
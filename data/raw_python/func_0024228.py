def get_pool_context(self):
        # TODO: Add in-process caching
        """
        Builds context for the WF pool.

        Returns:
            Context dict.
        """
        context = {self.current.lane_id: self.current.role, 'self': self.current.role}
        for lane_id, role_id in self.current.pool.items():
            if role_id:
                context[lane_id] = lazy_object_proxy.Proxy(
                    lambda: self.role_model(super_context).objects.get(role_id))
        return context
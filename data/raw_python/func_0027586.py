def load_entrypoint(self, entry_point_group):
        """Load entrypoint.

        :param entry_point_group: A name of entry point group used to load
            ``webassets`` bundles.

        .. versionchanged:: 1.0.0b2
           The *entrypoint* has been renamed to *entry_point_group*.
        """
        for ep in pkg_resources.iter_entry_points(entry_point_group):
            self.env.register(ep.name, ep.load())
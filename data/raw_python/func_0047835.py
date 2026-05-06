def _init_object(self, catalog_id, proxy, runtime, cat_name, cat_class):
        """Initialize this object as an OsidObject....do we need this??
        From the Mongo learning impl, but seems unnecessary for Handcar"""
        self._catalog_identifier = None
        self._init_proxy_and_runtime(proxy, runtime)
        self._catalog = cat_class(self._my_catalog_map)
        self._catalog._authority = self._authority  # there should be a better way...
        self._catalog_id = self._catalog.get_id()
        self._forms = dict()
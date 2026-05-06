def _init_object(self, catalog_id, proxy, runtime, db_name, cat_name, cat_class):
        """Initialize this session an OsidObject based session."""
        self._catalog_identifier = None
        self._init_proxy_and_runtime(proxy, runtime)

        uses_cataloging = False
        if catalog_id is not None and catalog_id.get_identifier() != PHANTOM_ROOT_IDENTIFIER:
            self._catalog_identifier = catalog_id.get_identifier()

            config = self._runtime.get_configuration()
            parameter_id = Id('parameter:' + db_name + 'CatalogingProviderImpl@mongo')

            try:
                provider_impl = config.get_value_by_parameter(parameter_id).get_string_value()
            except (AttributeError, KeyError, errors.NotFound):
                collection = JSONClientValidated(db_name,
                                                 collection=cat_name,
                                                 runtime=self._runtime)
                try:
                    self._my_catalog_map = collection.find_one({'_id': ObjectId(self._catalog_identifier)})
                except errors.NotFound:
                    if catalog_id.get_identifier_namespace() != db_name + '.' + cat_name:
                        self._my_catalog_map = self._create_orchestrated_cat(catalog_id, db_name, cat_name)
                    else:
                        raise errors.NotFound('could not find catalog identifier ' + catalog_id.get_identifier() + cat_name)
            else:
                uses_cataloging = True
                cataloging_manager = self._runtime.get_manager('CATALOGING',
                                                               provider_impl)  # need to add version argument
                lookup_session = cataloging_manager.get_catalog_lookup_session()
                # self._my_catalog_map = lookup_session.get_catalog(catalog_id)._my_map
                # self._catalog = Catalog(osid_object_map=self._my_catalog_map, runtime=self._runtime,
                #                         proxy=self._proxy)
                self._catalog = lookup_session.get_catalog(catalog_id)
        else:
            self._catalog_identifier = PHANTOM_ROOT_IDENTIFIER
            self._my_catalog_map = make_catalog_map(cat_name, identifier=self._catalog_identifier)

        if not uses_cataloging:
            self._catalog = cat_class(osid_object_map=self._my_catalog_map, runtime=self._runtime, proxy=self._proxy)

        self._catalog._authority = self._authority  # there should be a better way...
        self._catalog_id = self._catalog.get_id()
        self._forms = dict()
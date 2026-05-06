def _get_descendent_cat_idstrs(self, cat_id, hierarchy_session=None):
        """Recursively returns a list of all descendent catalog ids, inclusive"""
        def get_descendent_ids(h_session):
            idstr_list = [str(cat_id)]
            if h_session is None:
                pkg_name = cat_id.get_identifier_namespace().split('.')[0]
                cat_name = cat_id.get_identifier_namespace().split('.')[1]
                try:
                    mgr = self._get_provider_manager('HIERARCHY')
                    h_session = mgr.get_hierarchy_traversal_session_for_hierarchy(
                        Id(authority=pkg_name.upper(),
                           namespace='CATALOG',
                           identifier=cat_name.upper()),
                        proxy=self._proxy)
                except (errors.OperationFailed, errors.Unsupported):
                    return idstr_list  # there is no hierarchy
            if h_session.has_children(cat_id):
                for child_id in h_session.get_children(cat_id):
                    idstr_list += self._get_descendent_cat_idstrs(child_id, h_session)
            return list(set(idstr_list))

        use_caching = False
        try:
            config = self._runtime.get_configuration()
            parameter_id = Id('parameter:useCachingForQualifierIds@json')
            if config.get_value_by_parameter(parameter_id).get_boolean_value():
                use_caching = True
            else:
                pass
        except (AttributeError, KeyError, errors.NotFound):
            pass
        if use_caching:
            key = 'descendent-catalog-ids-{0}'.format(str(cat_id))

            # If configured to use memcache as the caching engine, use it.
            # Otherwise default to diskcache
            caching_engine = 'diskcache'

            try:
                config = self._runtime.get_configuration()
                parameter_id = Id('parameter:cachingEngine@json')
                caching_engine = config.get_value_by_parameter(parameter_id).get_string_value()
            except (AttributeError, KeyError, errors.NotFound):
                pass

            if caching_engine == 'memcache':
                import memcache
                caching_host = '127.0.0.1:11211'
                try:
                    config = self._runtime.get_configuration()
                    parameter_id = Id('parameter:cachingHostURI@json')
                    caching_host = config.get_value_by_parameter(parameter_id).get_string_value()
                except (AttributeError, KeyError, errors.NotFound):
                    pass

                mc = memcache.Client([caching_host], debug=0)

                catalog_ids = mc.get(key)
                if catalog_ids is None:
                    catalog_ids = get_descendent_ids(hierarchy_session)
                    mc.set(key, catalog_ids)
            elif caching_engine == 'diskcache':
                import diskcache
                with diskcache.Cache('/tmp/dlkit_cache') as cache:
                    # A little bit non-DRY, since it's almost the same as for memcache above.
                    # However, for diskcache.Cache, we have to call ".close()" or use a
                    #   ``with`` statement to safeguard calling ".close()", so we keep this
                    #   separate from the memcache implementation.
                    catalog_ids = cache.get(key)
                    if catalog_ids is None:
                        catalog_ids = get_descendent_ids(hierarchy_session)
                        cache.set(key, catalog_ids)
            else:
                raise errors.NotFound('The {0} caching engine was not found.'.format(caching_engine))
        else:
            catalog_ids = get_descendent_ids(hierarchy_session)
        return catalog_ids
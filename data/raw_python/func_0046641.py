def _get_parent_id_list(self, qualifier_id, hierarchy_id):
        """Returns list of parent id strings for qualifier_id in hierarchy.

        Uses memcache if caching is enabled.

        """
        if self._caching_enabled():
            key = 'parent_id_list_{0}'.format(str(qualifier_id))

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
                parent_id_list = mc.get(key)
                if parent_id_list is None:
                    parent_ids = self._get_hierarchy_session(hierarchy_id).get_parents(qualifier_id)
                    parent_id_list = [str(parent_id) for parent_id in parent_ids]
                    mc.set(key, parent_id_list)

            elif caching_engine == 'diskcache':
                import diskcache
                with diskcache.Cache('/tmp/dlkit_cache') as cache:
                    # A little bit non-DRY, since it's almost the same as for memcache above.
                    # However, for diskcache.Cache, we have to call ".close()" or use a
                    #   ``with`` statement to safeguard calling ".close()", so we keep this
                    #   separate from the memcache implementation.
                    parent_id_list = cache.get(key)
                    if parent_id_list is None:
                        parent_ids = self._get_hierarchy_session(hierarchy_id).get_parents(qualifier_id)
                        parent_id_list = [str(parent_id) for parent_id in parent_ids]
                        cache.set(key, parent_id_list)
            else:
                raise errors.NotFound('The {0} caching engine was not found.'.format(caching_engine))
        else:
            parent_ids = self._get_hierarchy_session(hierarchy_id).get_parents(qualifier_id)
            parent_id_list = [str(parent_id) for parent_id in parent_ids]
        return parent_id_list
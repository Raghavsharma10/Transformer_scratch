def _create_getters(self, klass):
        '''
        This method creates both the singular and plural getters for various
        Harvest object classes.

        '''
        flag_name = '_got_' + klass.element_name
        cache_name = '_' + klass.element_name
        setattr(self, cache_name, {})
        setattr(self, flag_name, False)
        cache = getattr(self, cache_name)

        def _get_item(id):
            if id in cache:
                return cache[id]
            else:
                url = '{}/{}'.format(klass.base_url, id)
                item = self._get_element_values(url, klass.element_name).next()
                item = klass(self, item)
                cache[id] = item
                return item

        setattr(self, klass.element_name, _get_item)

        def _get_items():
            if getattr(self, flag_name):
                for item in cache.values():
                    yield item
            else:
                for element in self._get_element_values(klass.base_url, klass.element_name):
                    item = klass(self, element)
                    cache[item.id] = item
                    yield item

                setattr(self, flag_name, True)

        setattr(self, klass.plural_name, _get_items)
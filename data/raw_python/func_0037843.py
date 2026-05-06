def _process_blacklist(self, blacklist):
        """
        Process blacklist into set of excluded versions
        """

        # Assume blacklist is correct format since it is checked by PluginLoader

        blacklist_cache = {}
        blacklist_cache_old = self._cache.get('blacklist', {})

        for entry in blacklist:

            blackkey = (entry.version, entry.operator)

            if blackkey in blacklist_cache:
                continue
            elif blackkey in blacklist_cache_old:
                blacklist_cache[blackkey] = blacklist_cache_old[blackkey]
            else:
                entry_cache = blacklist_cache[blackkey] = set()
                blackversion = parse_version(entry.version or '0')
                blackop = OPERATORS[entry.operator]

                for key in self:
                    if blackop(parse_version(key), blackversion):
                        entry_cache.add(key)

        self._cache['blacklist'] = blacklist_cache
        return set().union(*blacklist_cache.values())
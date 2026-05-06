def load_all_assistants(cls, superassistants):
        """Fills self._assistants with loaded YamlAssistant instances of requested roles.

        Tries to use cache (updated/created if needed). If cache is unusable, it
        falls back to loading all assistants.

        Args:
            roles: list of required assistant roles
        """
        # mapping of assistant roles to lists of top-level assistant instances
        _assistants = {}
        # {'crt': CreatorAssistant, ...}
        superas_dict = dict(map(lambda a: (a.name, a), superassistants))
        to_load = set(superas_dict.keys())
        for tl in to_load:
            dirs = [os.path.join(d, tl) for d in cls.assistants_dirs]
            file_hierarchy = cls.get_assistants_file_hierarchy(dirs)
            # load all if we're not using cache or if we fail to load it
            load_all = not settings.USE_CACHE
            if settings.USE_CACHE:
                try:
                    cch = cache.Cache()
                    cch.refresh_role(tl, file_hierarchy)
                    _assistants[tl] = cls.get_assistants_from_cache_hierarchy(cch.cache[tl],
                                                                                  superas_dict[tl],
                                                                                  role=tl)
                except BaseException as e:
                    logger.debug('Failed to use DevAssistant cachefile {0}: {1}'.format(
                        settings.CACHE_FILE, e))
                    load_all = True
            if load_all:
                _assistants[tl] = cls.get_assistants_from_file_hierarchy(file_hierarchy,
                                                                             superas_dict[tl],
                                                                             role=tl)
        return _assistants
def _sort_converters(cls, app_ready=False):
        '''Sorts the converter functions'''
        # app_ready is True when called from DMP's AppConfig.ready()
        # we can't sort before then because models aren't ready
        cls._sorting_enabled = cls._sorting_enabled or app_ready
        if cls._sorting_enabled:
            for converter in cls.converters:
                converter.prepare_sort_key()
            cls.converters.sort(key=attrgetter('sort_key'))
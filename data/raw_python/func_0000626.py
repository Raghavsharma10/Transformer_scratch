def __prepare_bloom(self):
        """Prepare bloom for existing checks
        """
        self.__bloom = pybloom_live.ScalableBloomFilter()
        columns = [getattr(self.__table.c, key) for key in self.__update_keys]
        keys = select(columns).execution_options(stream_results=True).execute()
        for key in keys:
            self.__bloom.add(tuple(key))
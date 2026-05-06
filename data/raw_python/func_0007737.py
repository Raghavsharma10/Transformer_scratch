def __instances(self):
        """ Cache instances, allowing generators to be used and reused. 
            This fills a cache as the generator gets emptied, eventually
            reading exclusively from the cache.
        """
        for instance in self.__instances_cache:
            yield instance
        for instance in self.__instances_original:
            self.__instances_cache.append(instance)
            yield instance
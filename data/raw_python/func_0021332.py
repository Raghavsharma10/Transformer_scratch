def process_query_conditionally(self, query):
        """
        Process a Query that is used within a lazy loader.

        (the process_query_conditionally() method is a SQLAlchemy
        hook invoked only within lazyload.)
        """
        if query._current_path:
            mapper, prop = query._current_path[-2:]
            for cls in mapper.class_.__mro__:
                k = (cls, prop.key)
                relationship_option = self._relationship_options.get(k)
                if relationship_option:
                    query._cache = relationship_option
                    break
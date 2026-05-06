def get_container(cls, scheduler):
        """
        Create temporary instance for helper functions
        """
        if scheduler in cls._container_cache:
            return cls._container_cache[scheduler]
        else:
            c = cls(scheduler)
            cls._container_cache[scheduler] = c
            return c
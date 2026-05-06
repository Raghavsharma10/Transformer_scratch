def unregister(self, condition_set):
        """
        Unregisters a condition set with the manager.

        >>> operator.unregister(condition_set) #doctest: +SKIP
        """
        if callable(condition_set):
            condition_set = condition_set()
        registry.pop(condition_set.get_id(), None)
        registry_by_namespace.pop(condition_set.get_namespace(), None)
def register(self, condition_set):
        """
        Registers a condition set with the manager.

        >>> condition_set = MyConditionSet() #doctest: +SKIP
        >>> operator.register(condition_set) #doctest: +SKIP
        """

        if callable(condition_set):
            condition_set = condition_set()
        registry[condition_set.get_id()] = condition_set
        registry_by_namespace[condition_set.get_namespace()] = condition_set
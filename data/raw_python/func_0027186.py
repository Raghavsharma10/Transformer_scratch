def get_configuration(cls, resource):
        """ Return how much consumables are used by resource with current configuration.

            Output example:
            {
                <ConsumableItem instance>: <usage>,
                <ConsumableItem instance>: <usage>,
                ...
            }
        """
        strategy = cls._get_strategy(resource.__class__)
        return strategy.get_configuration(resource)
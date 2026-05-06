def with_random_weights(cls, options):
        """
        Initialize from a list of options with random weights.

        The weights assigned to each object are uniformally random
        integers between ``1`` and ``len(options)``

        Args:
            options (list): The list of options of any type this object
                can return with the ``get()`` method.

        Returns:
            SoftOptions: A newly constructed instance
        """
        return cls([(value, random.randint(1, len(options)))
                    for value in options])
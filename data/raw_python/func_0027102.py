def fetcher_with_object(cls, parent_object, relationship="child"):
        """ Register the fetcher for a served object.

            This method will fill the fetcher with `managed_class` instances

            Args:
                parent_object: the instance of the parent object to serve

            Returns:
                It returns the fetcher instance.
        """

        fetcher = cls()
        fetcher.parent_object = parent_object
        fetcher.relationship = relationship

        rest_name = cls.managed_object_rest_name()
        parent_object.register_fetcher(fetcher, rest_name)

        return fetcher
def assign_methods(self, resource_class):
        """
        Given a resource_class and it's Meta.methods tuple,
        assign methods for communicating with that resource.

        Args:
            resource_class: A single resource class
        """
        assert all([
            x.upper() in VALID_METHODS for x in resource_class.Meta.methods])
        for method in resource_class.Meta.methods:

            self._assign_method(
                resource_class,
                method.upper()
            )
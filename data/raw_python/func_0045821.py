def bind_to_instance(self, instance):
        """
        Bind a ResourceApi instance to an operation.
        """
        self.binding = instance
        self.middleware.append(instance)
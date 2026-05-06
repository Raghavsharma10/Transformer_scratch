def to_swagger(self, bound_resource=None):
        """
        Generate a swagger representation.
        """
        return _to_swagger(
            {
                'name': self.name,
                'in': self.in_.value,
                'type': str(self.type) if self.type else None,
            },
            description=self.description,
            resource=bound_resource if self.resource is DefaultResource else self.resource,
            options=self.options
        )
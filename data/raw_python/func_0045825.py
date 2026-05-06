def to_swagger(self):
        """
        Generate a dictionary for documentation generation.
        """
        return dict_filter(
            operationId=self.operation_id,
            description=(self.callback.__doc__ or '').strip() or None,
            summary=self.summary or None,
            tags=list(self.tags) or None,
            deprecated=self.deprecated or None,
            consumes=list(self.consumes) or None,
            parameters=[param.to_swagger(self.resource) for param in self.parameters] or None,
            produces=list(self.produces) or None,
            responses=dict(resp.to_swagger(self.resource) for resp in self.responses) or None,
            security=self.security.to_swagger() if self.security else None,
        )
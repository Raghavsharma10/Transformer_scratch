def rendered(self):
        """The rendered wire format for all conditions that have been rendered.  Rendered conditions are never
        cleared.  A new :class:`~bloop.conditions.ConditionRenderer` should be used for each operation."""
        expressions = {k: v for (k, v) in self.expressions.items() if v is not None}
        if self.refs.attr_names:
            expressions["ExpressionAttributeNames"] = self.refs.attr_names
        if self.refs.attr_values:
            expressions["ExpressionAttributeValues"] = self.refs.attr_values
        return expressions
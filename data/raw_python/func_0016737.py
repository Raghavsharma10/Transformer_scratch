def render(self, obj=None, condition=None, atomic=False, update=False, filter=None, projection=None, key=None):
        """Main entry point for rendering multiple expressions.  All parameters are optional, except obj when
        atomic or update are True.

        :param obj: *(Optional)* An object to render an atomic condition or update expression for.  Required if
            update or atomic are true.  Default is False.
        :param condition: *(Optional)* Rendered as a "ConditionExpression" for a conditional operation.
            If atomic is True, the two are rendered in an AND condition.  Default is None.
        :type condition: :class:`~bloop.conditions.BaseCondition`
        :param bool atomic: *(Optional)*  True if an atomic condition should be created for ``obj`` and rendered as
            a "ConditionExpression".  Default is False.
        :param bool update: *(Optional)*  True if an "UpdateExpression" should be rendered for ``obj``.
            Default is False.
        :param filter: *(Optional)* A filter condition for a query or scan, rendered as a "FilterExpression".
            Default is None.
        :type filter: :class:`~bloop.conditions.BaseCondition`
        :param projection: *(Optional)* A set of Columns to include in a query or scan, redered as a
            "ProjectionExpression".  Default is None.
        :type projection: set :class:`~bloop.models.Column`
        :param key: *(Optional)* A key condition for queries, rendered as a "KeyConditionExpression".  Default is None.
        :type key: :class:`~bloop.conditions.BaseCondition`
        """
        if (atomic or update) and not obj:
            raise InvalidCondition("An object is required to render atomic conditions or updates without an object.")

        if filter:
            self.render_filter_expression(filter)

        if projection:
            self.render_projection_expression(projection)

        if key:
            self.render_key_expression(key)

        # Condition requires a bit of work, because either one can be empty/false
        condition = (condition or Condition()) & (get_snapshot(obj) if atomic else Condition())
        if condition:
            self.render_condition_expression(condition)

        if update:
            self.render_update_expression(obj)
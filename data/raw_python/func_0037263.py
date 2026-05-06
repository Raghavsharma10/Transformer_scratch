def has_active_condition(self, condition, instances):
        """
        Given a list of instances, and the condition active for
        this switch, returns a boolean representing if the
        conditional is met, including a non-instance default.
        """
        return_value = None
        for instance in instances + [None]:
            if not self.can_execute(instance):
                continue
            result = self.is_active(instance, condition)
            if result is False:
                return False
            elif result is True:
                return_value = True
        return return_value
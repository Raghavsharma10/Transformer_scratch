def _get_average_time_stamp(action_set):
        """Return the average time stamp for the rules in this action
        set."""
        # This is the average value of the iteration counter upon the most
        # recent update of each rule in this action set.
        total_time_stamps = sum(rule.time_stamp * rule.numerosity
                                for rule in action_set)
        total_numerosity = sum(rule.numerosity for rule in action_set)
        return total_time_stamps / (total_numerosity or 1)
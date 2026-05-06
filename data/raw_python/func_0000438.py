def get_stats(self, obj, stat_name):
        """ Send CLI command that returns list of integer counters.

        :param obj: requested object.
        :param stat_name: statistics command name.
        :return: list of counters.
        :rtype: list(int)
        """
        return [int(v) for v in self.send_command_return(obj, stat_name, '?').split()]
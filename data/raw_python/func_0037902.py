def get_stat(self, obj_name, stat_name):
        """
        :param obj_name: requested object name.
        :param stat_name: requested statistics name.
        :return: str, the value of the requested statics for the requested object.
        """

        return self.statistics[obj_name][self.captions.index(stat_name)]
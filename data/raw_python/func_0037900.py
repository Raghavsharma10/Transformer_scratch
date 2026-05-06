def get_object_stats(self, obj_name):
        """
        :param obj_name: requested object name
        :returns: all statistics values for the requested object.
        """

        return dict(zip(self.captions, self.statistics[obj_name]))
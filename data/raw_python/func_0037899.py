def get_all_stats(self):
        """
        :returns: all statistics values for all objects.
        """

        all_stats = OrderedDict()
        for obj_name in self.statistics:
            all_stats[obj_name] = dict(zip(self.captions, self.statistics[obj_name]))
        return all_stats
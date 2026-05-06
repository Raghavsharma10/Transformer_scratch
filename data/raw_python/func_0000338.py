def read_stats(self):
        """
        :return: dictionary {group name {stat name: value}}.
            Sea XenaTpld.stats_captions.
        """

        stats_with_captions = OrderedDict()
        for stat_name in self.stats_captions.keys():
            stats_with_captions[stat_name] = self.read_stat(self.stats_captions[stat_name], stat_name)
        return stats_with_captions
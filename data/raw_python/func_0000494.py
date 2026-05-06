def get_flat_stats(self):
        """
        :return: statistics as flat table {port/strea,/tpld name {group_stat name: value}}
        """
        flat_stats = OrderedDict()
        for obj, port_stats in self.statistics.items():
            flat_obj_stats = OrderedDict()
            for group_name, group_values in port_stats.items():
                for stat_name, stat_value in group_values.items():
                    full_stat_name = group_name + '_' + stat_name
                    flat_obj_stats[full_stat_name] = stat_value
            flat_stats[obj.name] = flat_obj_stats
        return flat_stats
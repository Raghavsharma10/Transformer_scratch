def from_dict(cls, d):
        """Used to create an instance of this class from a pstats dict item"""
        stats = []
        for (filename, lineno, name), stat_values in d.iteritems():
            if len(stat_values) == 5:
                ncalls, ncall_nr, total_time, cum_time, subcall_stats = stat_values
            else:
                ncalls, ncall_nr, total_time, cum_time = stat_values
                subcall_stats = None
            stat = cProfileFuncStat(filename, lineno, name, ncalls, ncall_nr, total_time, cum_time, subcall_stats)
            stats.append(stat)

        return stats
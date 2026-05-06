def from_json(cls, filename):
        """
        Imports statistical data from a JSON formatted file

        Parameters
        ----------
        filename:    input file that holds statistics data
        """
        def json_decoder(d):
            if 'p01' in d and 'pxx' in d:  # we assume this is a CascadeStatistics object
                return melodist.cascade.CascadeStatistics.from_dict(d)

            return d

        with open(filename) as f:
            d = json.load(f, object_hook=json_decoder)

        stats = cls()

        stats.temp.update(d['temp'])
        stats.hum.update(d['hum'])
        stats.precip.update(d['precip'])
        stats.wind.update(d['wind'])
        stats.glob.update(d['glob'])

        if stats.temp.max_delta is not None:
            stats.temp.max_delta = pd.read_json(json.dumps(stats.temp.max_delta), typ='series').sort_index()

        if stats.temp.mean_course is not None:
            mc = pd.read_json(json.dumps(stats.temp.mean_course), typ='frame').sort_index()[np.arange(1, 12 + 1)]
            stats.temp.mean_course = mc.sort_index()[np.arange(1, 12 + 1)]

        if stats.hum.month_hour_precip_mean is not None:
            mhpm = pd.read_json(json.dumps(stats.hum.month_hour_precip_mean), typ='frame').sort_index()
            mhpm = mhpm.set_index(['level_0', 'level_1', 'level_2'])  # convert to MultiIndex
            mhpm = mhpm.squeeze()  # convert to Series
            mhpm = mhpm.rename_axis([None, None, None])  # remove index labels
            stats.hum.month_hour_precip_mean = mhpm

        for var in ('angstroem', 'bristcamp', 'mean_course'):
            if stats.glob[var] is not None:
                stats.glob[var] = pd.read_json(json.dumps(stats.glob[var])).sort_index()

        if stats.glob.mean_course is not None:
            stats.glob.mean_course = stats.glob.mean_course[np.arange(1, 12 + 1)]

        return stats
def clean(self, algorithm=None):
        '''Create a new :class:`TimeSeries` with missing data removed or
replaced by the *algorithm* provided'''
        # all dates
        original_dates = list(self.dates())
        series = []
        all_dates = set()
        for serie in self.series():
            dstart, dend, vend = None, None, None
            new_dates = []
            new_values = []
            missings = []
            values = {}
            for d, v in zip(original_dates, serie):
                if v == v:
                    if dstart is None:
                        dstart = d
                    if missings:
                        for dx, vx in algorithm(dend, vend, d, v, missings):
                            new_dates.append(dx)
                            new_values.append(vx)
                        missings = []
                    dend = d
                    vend = v
                    values[d] = v
                elif dstart is not None and algorithm:
                    missings.append((dt, v))
            if missings:
                for dx, vx in algorithm(dend, vend, None, None, missings):
                    new_dates.append(dx)
                    new_values.append(vx)
                    dend = dx
            series.append((dstart, dend, values))
            all_dates = all_dates.union(values)
        cdate = []
        cdata = []
        for dt in sorted(all_dates):
            cross = []
            for start, end, values in series:
                if start is None or (dt >= start and dt <= end):
                    value = values.get(dt)
                    if value is None:
                        cross = None
                        break
                else:
                    value = nan
                cross.append(value)
            if cross:
                cdate.append(dt)
                cdata.append(cross)
        return self.clone(date=cdate, data=cdata)
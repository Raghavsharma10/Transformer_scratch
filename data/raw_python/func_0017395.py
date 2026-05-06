def named_series(self, ordering=None):
        '''Generator of tuples with name and serie data.'''
        series = self.series()
        if ordering:
            series = list(series)
            todo = dict(((n, idx) for idx, n in enumerate(self.names())))
            for name in ordering:
                if name in todo:
                    idx = todo.pop(name)
                    yield name, series[idx]
            for name in todo:
                idx = todo[name]
                yield name, series[idx]
        else:
            for name_serie in zip(self.names(), series):
                yield name_serie
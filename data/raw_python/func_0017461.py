def names(self, with_namespace=False):
        '''List of names for series in dataset.

        It will always return a list or names with length given by
        :class:`~.DynData.count`.
        '''
        N = self.count()
        names = self.name.split(settings.splittingnames)[:N]
        n = 0
        while len(names) < N:
            n += 1
            names.append('unnamed%s' % n)
        if with_namespace and self.namespace:
            n = self.namespace
            s = settings.field_separator
            return [n + s + f for f in names]
        else:
            return names
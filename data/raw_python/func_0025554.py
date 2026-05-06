def signature(self):
        "Instance file name"
        kw = self.get_params()
        keys = sorted(kw.keys())
        l = []
        for k in keys:
            n = k[0] + k[-1]
            v = kw[k]
            if k == 'function_set':
                v = "_".join([x.__name__[0] +
                              x.__name__[-1] +
                              str(x.nargs) for x in kw[k]])
            elif k == 'population_class':
                v = kw[k].__name__
            else:
                v = str(v)
            l.append('{0}_{1}'.format(n, v))
        return '-'.join(l)
def paramsReport(self):
        """See docs for `Model` abstract base class."""
        report = {}
        for param in self._REPORTPARAMS:
            pvalue = getattr(self, param)
            if isinstance(pvalue, float):
                report[param] = pvalue
            elif isinstance(pvalue, scipy.ndarray) and pvalue.shape == (3, N_NT):
                for p in range(3):
                    for w in range(N_NT - 1):
                        report['{0}{1}{2}'.format(param, p, INDEX_TO_NT[w])] =\
                                pvalue[p][w]
            else:
                raise ValueError("Unexpected param: {0}".format(param))
        return report
def decision_function(self, X, **kwargs):
        "Decision function i.e. the raw data of the prediction"
        if X is None:
            return self._hy_test
        X = self.convert_features(X)
        if len(X) < self.nvar:
            _ = 'Number of variables differ, trained with %s given %s' % (self.nvar, len(X))
            raise RuntimeError(_)
        hist = self._hist
        for node in hist:
            if node.height:
                node.eval(hist)
            else:
                node.eval(X)
        node.normalize()
        r = node.hy
        for i in hist[:-1]:
            i.hy = None
            i.hy_test = None
        gc.collect()
        return r
def predict(self, v=None, X=None):
        """In classification this returns the classes, in
        regression it is equivalent to the decision function"""
        if X is None:
            X = v
            v = None
        m = self.model(v=v)
        return m.predict(X)
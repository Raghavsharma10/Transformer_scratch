def get_pres_features(self, features=None):
        """
        Returns a df of features for presented items
        """
        if features is None:
            features = self.dist_funcs.keys()
        elif not isinstance(features, list):
            features = [features]
        return self.pres.applymap(lambda x: {k:v for k,v in x.items() if k in features} if x is not None else None)
def _rel(self, copy=False):
        """
        Get descriptive kwargs of the container (e.g. name, description, meta).
        """
        rel = {}
        for key, obj in vars(self).items():
            if not isinstance(obj, (pd.Series, pd.DataFrame, pd.SparseSeries, pd.SparseDataFrame)) and not key.startswith('_'):
                if copy and 'id' not in key:
                    rel[key] = deepcopy(obj)
                else:
                    rel[key] = obj
        return rel
def _bool_encode(self, d):
        """
        Converts bool values to lowercase strings
        
        """
        for k, v in d.items():
            if isinstance(v, bool):
                d[k] = str(v).lower()
        
        return d
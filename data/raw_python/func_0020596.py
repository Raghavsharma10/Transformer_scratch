def _utf8_encode(self, d):
        """
        Ensures all values are encoded in UTF-8 and converts them to lowercase
        
        """
        for k, v in d.items():
            if isinstance(v, str):
                d[k] = v.encode('utf8').lower()
            if isinstance(v, list):
                for index,item in enumerate(v):
                    item = item.encode('utf8').lower()
                    v[index] = item
            if isinstance(v, dict):
                d[k] = self._utf8_encode(v)
        
        return d
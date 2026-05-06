def set_nested(self, klist, value):
        """D.set_nested((k1, k2,k3, ...), v) -> D[k1][k2][k3] ... = v"""
        keys = list(klist)
        if len(keys) > 0:
            curr_dict = self
            last_key = keys.pop()
            for key in keys:
                if not curr_dict.has_key(key) or not isinstance(curr_dict[key], 
                                                                NestedDict):
                    curr_dict[key] = type(self)()
                curr_dict = curr_dict[key]
            curr_dict[last_key] = value
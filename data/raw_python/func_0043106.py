def leaves(self, prefix=None):
        """
        LIKE items() BUT RECURSIVE, AND ONLY FOR THE LEAVES (non dict) VALUES
        """
        prefix = coalesce(prefix, "")
        output = []
        for k, v in self.items():
            if _get(v, CLASS) in data_types:
                output.extend(wrap(v).leaves(prefix=prefix + literal_field(k) + "."))
            else:
                output.append((prefix + literal_field(k), v))
        return output
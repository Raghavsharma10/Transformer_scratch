def to_dict(self, in_dict=None):
        """
        Turn the Namespace and sub Namespaces back into a native
        python dictionary.

        :param in_dict: Do not use, for self recursion
        :return: python dictionary of this Namespace
        """
        in_dict = in_dict if in_dict else self
        out_dict = dict()
        for k, v in in_dict.items():
            if isinstance(v, Namespace):
                v = v.to_dict()
            out_dict[k] = v
        return out_dict
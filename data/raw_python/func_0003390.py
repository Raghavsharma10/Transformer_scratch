def todict(self):
        """
        Convert this node to a dictionary tree.
        """
        dict_entry = []
        for k,v in self.items():
            if isinstance(v, ConfigTree):
                dict_entry.append((k, v.todict()))
            else:
                dict_entry.append((k, v))
        return dict(dict_entry)
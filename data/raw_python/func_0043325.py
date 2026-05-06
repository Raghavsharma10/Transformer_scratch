def to_list(self):
        """
        To a list of dicts (each dict is an instances)
        """
        ret = []
        for instance in self.instances:
            ret.append(instance.to_dict())
        return ret
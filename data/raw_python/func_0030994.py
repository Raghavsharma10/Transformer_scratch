def as_named_tuple(self):
        """
        Returns the File as a named tuple. The full path plus all entity
        key/value pairs are returned as attributes.
        """
        keys = list(self.entities.keys())
        replaced = []
        for i, k in enumerate(keys):
            if iskeyword(k):
                replaced.append(k)
                keys[i] = '%s_' % k
        if replaced:
            safe = ['%s_' % k for k in replaced]
            warnings.warn("Entity names cannot be reserved keywords when "
                          "representing a File as a namedtuple. Replacing "
                          "entities %s with safe versions %s." % (keys, safe))
        entities = dict(zip(keys, self.entities.values()))
        _File = namedtuple('File', 'filename ' + ' '.join(entities.keys()))
        return _File(filename=self.path, **entities)
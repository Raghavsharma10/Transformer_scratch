def title(self):
        """
        get title of this node. If an entry for this course is found in the configuration namemap it is used, otherwise the default
        value from stud.ip is used.
        """
        tmp = c.namemap_lookup(self.id) if c.namemap_lookup(self.id) is not None else self._title
        return secure_filename(tmp)
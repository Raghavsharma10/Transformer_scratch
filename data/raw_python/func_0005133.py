def format(self):
        """ Formats the __repr__ string
            -> #str containing __repr__ output
        """
        _bold = bold
        _break = "\n    "
        if not self.pretty:
            _bold = lambda x: x
        # Attach memory address and return
        _attrs = self._format_attrs()
        parent_name = get_parent_name(self.obj) if self.full_name else None
        self.data = "<{}{}:{}{}>{}".format(
            parent_name + "." if parent_name else "",
            get_obj_name(self.obj),
            _attrs,
            ":{}".format(hex(id(self.obj))) if self.address else "",
            _break+self.supplemental if self.supplemental else "")
        return stdout_encode(self.data)
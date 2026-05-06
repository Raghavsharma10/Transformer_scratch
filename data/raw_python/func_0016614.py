def _read_header(self):
        """
        When a CSV file is given, extracts header information the file.
        Otherwise, this header data must be explicitly given when the object
        is instantiated.
        """
        if not self.filename or self.header_types:
            return
        rows = csv.reader(open(self.filename))
        #header = rows.next()
        header = next(rows)
        self.header_types = {} # {attr_name:type}
        self._class_attr_name = None
        self.header_order = [] # [attr_name,...]
        for el in header:
            matches = ATTR_HEADER_PATTERN.findall(el)
            assert matches, "Invalid header element: %s" % (el,)
            el_name, el_type, el_mode = matches[0]
            el_name = el_name.strip()
            self.header_order.append(el_name)
            self.header_types[el_name] = el_type
            if el_mode == ATTR_MODE_CLASS:
                assert self._class_attr_name is None, \
                    "Multiple class attributes are not supported."
                self._class_attr_name = el_name
            else:
                assert self.header_types[el_name] != ATTR_TYPE_CONTINUOUS, \
                    "Non-class continuous attributes are not supported."
        assert self._class_attr_name, "A class attribute must be specified."
def update_rec(self, rec, name, value):
        """Update current GOTerm with optional record."""
        # 'def' is a reserved word in python, do not use it as a Class attr.
        if name == "def":
            name = "defn"

        # If we have a relationship, then we will split this into a further
        # dictionary.

        if hasattr(rec, name):
            if name not in self.attrs_scalar:
                if name not in self.attrs_nested:
                    getattr(rec, name).add(value)
                else:
                    self._add_nested(rec, name, value)
            else:
                raise Exception("ATTR({NAME}) ALREADY SET({VAL})".format(
                    NAME=name, VAL=getattr(rec, name)))
        else: # Initialize new GOTerm attr
            if name in self.attrs_scalar:
                setattr(rec, name, value)
            elif name not in self.attrs_nested:
                setattr(rec, name, set([value]))
            else:
                name = '_{:s}'.format(name)
                setattr(rec, name, defaultdict(list))
                self._add_nested(rec, name, value)
def unpack(self):
        """Decompose a GVariant into a native Python object."""

        LEAF_ACCESSORS = {
            'b': self.get_boolean,
            'y': self.get_byte,
            'n': self.get_int16,
            'q': self.get_uint16,
            'i': self.get_int32,
            'u': self.get_uint32,
            'x': self.get_int64,
            't': self.get_uint64,
            'h': self.get_handle,
            'd': self.get_double,
            's': self.get_string,
            'o': self.get_string,  # object path
            'g': self.get_string,  # signature
        }

        # simple values
        la = LEAF_ACCESSORS.get(self.get_type_string())
        if la:
            return la()

        # tuple
        if self.get_type_string().startswith('('):
            res = [self.get_child_value(i).unpack()
                   for i in range(self.n_children())]
            return tuple(res)

        # dictionary
        if self.get_type_string().startswith('a{'):
            res = {}
            for i in range(self.n_children()):
                v = self.get_child_value(i)
                res[v.get_child_value(0).unpack()] = v.get_child_value(1).unpack()
            return res

        # array
        if self.get_type_string().startswith('a'):
            return [self.get_child_value(i).unpack()
                    for i in range(self.n_children())]

        # variant (just unbox transparently)
        if self.get_type_string().startswith('v'):
            return self.get_variant().unpack()

        # maybe
        if self.get_type_string().startswith('m'):
            m = self.get_maybe()
            return m.unpack() if m else None

        raise NotImplementedError('unsupported GVariant type ' + self.get_type_string())
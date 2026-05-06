def tdController(self):
        """Get the next controller while iterating.

        :return: a dict with the keys: id, type, name, available.
        """
        cid = c_int()
        ctype = c_int()
        name = create_string_buffer(255)
        available = c_int()

        self._lib.tdController(byref(cid), byref(ctype), name, sizeof(name),
                               byref(available))
        return {'id': cid.value, 'type': ctype.value,
                'name': self._to_str(name), 'available': available.value}
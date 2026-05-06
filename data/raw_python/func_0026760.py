def tdSensor(self):
        """Get the next sensor while iterating.

        :return: a dict with the keys: protocol, model, id, datatypes.
        """
        protocol = create_string_buffer(20)
        model = create_string_buffer(20)
        sid = c_int()
        datatypes = c_int()

        self._lib.tdSensor(protocol, sizeof(protocol), model, sizeof(model),
                           byref(sid), byref(datatypes))
        return {'protocol': self._to_str(protocol),
                'model': self._to_str(model),
                'id': sid.value, 'datatypes': datatypes.value}
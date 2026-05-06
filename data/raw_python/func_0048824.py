def _init_map(self):
        """stub"""
        super(IRTItemFormRecord, self)._init_map()
        self.my_osid_object_form._my_map['decimalValues']['difficulty'] = \
            self._decimal_value_metadata['default_decimal_values'][1]
        self.my_osid_object_form._my_map['decimalValues']['discrimination'] = \
            self._decimal_value_metadata['default_decimal_values'][1]
        self.my_osid_object_form._my_map['decimalValues']['pseudoGuessing'] = \
            self._decimal_value_metadata['default_decimal_values'][1]
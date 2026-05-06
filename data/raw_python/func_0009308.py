def set_value(self, comp_str):
        """
        Set the value of component.

        :param string comp_str: value of component
        :returns: None
        :exception: ValueError - incorrect value of component
        """

        self._is_negated = False
        self._encoded_value = comp_str
        self._standard_value = super(
            CPEComponent2_3_URI_edpacked, self)._decode()
def set_value(self, comp_str, comp_att):
        """
        Set the value of component.

        :param string comp_str: value of component
        :param string comp_att: attribute associated with comp_str
        :returns: None
        :exception: ValueError - incorrect value of component
        """

        # Del double quotes of value
        str = comp_str[1:-1]
        self._standard_value = str

        # Parse the value
        super(CPEComponent2_3_WFN, self).set_value(str, comp_att)
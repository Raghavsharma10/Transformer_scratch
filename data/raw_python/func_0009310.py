def _is_valid_value(self):
        """
        Return True if the value of component in generic attribute is valid,
        and otherwise False.

        :returns: True if value is valid, False otherwise
        :rtype: boolean
        """

        comp_str = self._encoded_value

        value_pattern = []
        value_pattern.append("^((")
        value_pattern.append("~[")
        value_pattern.append(CPEComponent1_1._STRING)
        value_pattern.append("]+")
        value_pattern.append(")|(")
        value_pattern.append("[")
        value_pattern.append(CPEComponent1_1._STRING)
        value_pattern.append("]+(![")
        value_pattern.append(CPEComponent1_1._STRING)
        value_pattern.append("]+)*")
        value_pattern.append("))$")

        value_rxc = re.compile("".join(value_pattern))
        return value_rxc.match(comp_str) is not None
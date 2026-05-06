def _is_valid_edition(self):
        """
        Return True if the input value of attribute "edition" is valid,
        and otherwise False.

        :returns: True if value is valid, False otherwise
        :rtype: boolean
        """

        comp_str = self._standard_value[0]

        packed = []
        packed.append("(")
        packed.append(CPEComponent2_3_URI.SEPARATOR_PACKED_EDITION)
        packed.append(CPEComponent2_3_URI._string)
        packed.append("){5}")

        value_pattern = []
        value_pattern.append("^(")
        value_pattern.append(CPEComponent2_3_URI._string)
        value_pattern.append("|")
        value_pattern.append("".join(packed))
        value_pattern.append(")$")

        value_rxc = re.compile("".join(value_pattern))
        return value_rxc.match(comp_str) is not None
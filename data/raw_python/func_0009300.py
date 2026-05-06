def _is_valid_part(self):
        """
        Return True if the value of component in attribute "part" is valid,
        and otherwise False.

        :returns: True if value of component is valid, False otherwise
        :rtype: boolean
        """

        comp_str = self._encoded_value.lower()
        part_rxc = re.compile(CPEComponentSimple._PART_PATTERN)
        return part_rxc.match(comp_str) is not None
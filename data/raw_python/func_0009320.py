def _is_valid_part(self):
        """
        Return True if the value of component in attribute "part" is valid,
        and otherwise False.

        :returns: True if value of component is valid, False otherwise
        :rtype: boolean
        """

        comp_str = self._encoded_value

        # Check if value of component do not have wildcard
        if ((comp_str.find(self.WILDCARD_ONE) == -1) and
           (comp_str.find(self.WILDCARD_MULTI) == -1)):

            return super(CPEComponent2_3, self)._is_valid_part()

        # Compilation of regular expression associated with value of part
        part_pattern = "^(\{0}|\{1})$".format(self.WILDCARD_ONE,
                                              self.WILDCARD_MULTI)
        part_rxc = re.compile(part_pattern)

        return part_rxc.match(comp_str) is not None
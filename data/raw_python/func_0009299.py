def _is_valid_language(self):
        """
        Return True if the value of component in attribute "language" is valid,
        and otherwise False.

        :returns: True if value is valid, False otherwise
        :rtype: boolean
        """

        comp_str = self._encoded_value.lower()
        lang_rxc = re.compile(CPEComponentSimple._LANGTAG_PATTERN)
        return lang_rxc.match(comp_str) is not None
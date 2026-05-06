def filter_chars(self, chars=u""):
        """
        Return a new IPAString, containing only the IPA characters specified
        by the ``chars`` string.

        Valid values for ``chars`` are:

        * ``consonants`` or ``cns``
        * ``vowels`` or ``vwl``
        * ``letters`` or ``cns_vwl``
        * ``cns_vwl_pstr`` or ``cvp``
        * ``cns_vwl_pstr_long`` or ``cvpl``
        * ``cns_vwl_str`` or ``cvs``
        * ``cns_vwl_str_len`` or ``cvsl``
        * ``cns_vwl_str_len_wb`` or ``cvslw``
        * ``cns_vwl_str_len_wb_sb`` or ``cvslws``

        :rtype: IPAString
        """
        if chars in [u"cns", u"consonants"]:
            return self.consonants
        elif chars in [u"vwl", u"vowels"]:
            return self.vowels
        elif chars in [u"cns_vwl", u"letters"]:
            return self.letters
        elif chars in [u"cns_vwl_pstr", u"cvp"]:
            return self.cns_vwl_pstr
        elif chars in [u"cns_vwl_pstr_long", u"cvpl"]:
            return self.cns_vwl_pstr_long
        elif chars in [u"cns_vwl_str", u"cvs"]:
            return self.cns_vwl_str
        elif chars in [u"cns_vwl_str_len", u"cvsl"]:
            return self.cns_vwl_str_len
        elif chars in [u"cns_vwl_str_len_wb", u"cvslw"]:
            return self.cns_vwl_str_len_wb
        elif chars in [u"cns_vwl_str_len_wb_sb", u"cvslws"]:
            return self.cns_vwl_str_len_wb_sb
        return self
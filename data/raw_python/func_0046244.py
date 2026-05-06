def xform(self, left, right, repeating, base, sign):
        """
        Return prefixes for tuple.

        :param str left: left of the radix
        :param str right: right of the radix
        :param str repeating: repeating part
        :param int base: the base in which value is displayed
        :param int sign: -1, 0, 1 as appropriate
        :returns: the number string
        :rtype: str
        """
        # pylint: disable=too-many-arguments

        base_prefix = ''
        if self.CONFIG.use_prefix:
            if base == 8:
                base_prefix = '0'
            elif base == 16:
                base_prefix = '0x'
            else:
                base_prefix = ''

        base_subscript = str(base) if self.CONFIG.use_subscript else ''

        result = {
           'sign' : '-' if sign == -1 else '',
           'base_prefix' : base_prefix,
           'left' : left,
           'radix' : '.' if (right != "" or repeating != "") else "",
           'right' : right,
           'repeating' : ("(%s)" % repeating) if repeating != "" else "",
           'base_separator' : '' if base_subscript == '' else '_',
           'base_subscript' : base_subscript
        }

        return self._FMT_STR % result
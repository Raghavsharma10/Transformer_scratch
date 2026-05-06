def _escape_identifiers(self, item):
        """
        This function escapes column and table names
        @param item:
        """
        if self._escape_char == '':
            return item

        for field in self._reserved_identifiers:
            if item.find('.%s' % field) != -1:
                _str = "%s%s" % (self._escape_char, item.replace('.', '%s.' % self._escape_char))
                # remove duplicates if the user already included the escape
                return re.sub(r'[%s]+'%self._escape_char, self._escape_char, _str)

        if item.find('.') != -1:
            _str = "%s%s%s" % (self._escape_char, item.replace('.', '%s.%s'%(self._escape_char, self._escape_char)),
            self._escape_char)
        else:
            _str = self._escape_char+item+self._escape_char
        # remove duplicates if the user already included the escape
        return re.sub(r'[%s]+'%self._escape_char, self._escape_char, _str)
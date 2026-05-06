def match(self, text, noprefix=False):
        """Matches date/datetime string against date patterns and returns pattern and parsed date if matched.
        It's not indeded for common usage, since if successful it returns date as array of numbers and pattern
        that matched this date

        :param text:
            Any human readable string
        :type date_string: str|unicode
        :param noprefix:
            If set True than doesn't use prefix based date patterns filtering settings
        :type noprefix: bool


        :return: Returns dicts with `values` as array of representing parsed date and 'pattern' with info about matched pattern if successful, else returns None
        :rtype: :class:`dict`."""
        n = len(text)
        if self.cachedpats is not None:
            pats = self.cachedpats
        else:
            pats = self.patterns
        if n > 5 and not noprefix:
            basekeys = self.__matchPrefix(text[:6])
        else:
            basekeys = []
        for p in pats:
            if n < p['length']['min'] or n > p['length']['max']: continue
            if p['right'] and len(basekeys) > 0 and p['basekey'] not in basekeys: continue
            try:
                r = p['pattern'].parseString(text)
                # Do sanity check
                d = r.asDict()
                if 'month' in d:
                    val = int(d['month'])
                    if val > 12 or val < 1:
                        continue
                if 'day' in d:
                    val = int(d['day'])
                    if val > 31 or val < 1:
                        continue
                return {'values' : r, 'pattern' : p}
            except ParseException as e:
#                print p['key'], text.encode('utf-8'), e
                pass
        return None
def match(self, item):
        '''Return whether *item* matches this collection expression.

        If a match is successful return data about the match otherwise return
        None.

        '''
        match = self._expression.match(item)
        if not match:
            return None

        index = match.group('index')
        padded = False
        if match.group('padding'):
            padded = True

        if self.padding == 0:
            if padded:
                return None

        elif len(index) != self.padding:
            return None

        return match
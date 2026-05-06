def _update_expression(self):
        '''Update internal expression.'''
        self._expression = re.compile(
            '^{0}(?P<index>(?P<padding>0*)\d+?){1}$'
            .format(re.escape(self.head), re.escape(self.tail))
        )
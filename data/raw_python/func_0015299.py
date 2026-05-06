def parse(self, expression):
        """
        Evaluates 'expression' and returns it's value(s)
        """
        if isinstance(expression, (list, dict)):
            return (True if expression else False, expression)
        if sys.version_info[0] > 2:
            self.next = self.tokenize(expression).__next__
        else:
            self.next = self.tokenize(expression).next
        self.token = self.next()
        return self.expression()
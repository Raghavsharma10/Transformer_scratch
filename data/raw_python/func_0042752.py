def reset_indentation(self, amount):
        """Replace previous indentation with desired amount"""
        while self.result and self.result[-1][0] == INDENT:
            self.result.pop()
        self.result.append((INDENT, amount))
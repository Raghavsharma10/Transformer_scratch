def compare_balance(self, operator, or_equals, amount):
        """Additional step using regex matcher to compare the current balance with some number"""
        amount = int(amount)
        if operator == 'less':
            if or_equals:
                self.assertLessEqual(self.balance, amount)
            else:
                self.assertLess(self.balance, amount)
        elif or_equals:
            self.assertGreaterEqual(self.balance, amount)
        else:
            self.assertGreater(self.balance, amount)
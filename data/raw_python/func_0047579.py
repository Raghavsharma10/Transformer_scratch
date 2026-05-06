def withdraw(self, amount):
        """Extends withdraw method to make sure enough funds are in the account, then call withdraw from superclass"""
        if amount > self.balance:
            raise ValueError('Insufficient Funds')
        super().withdraw(amount)
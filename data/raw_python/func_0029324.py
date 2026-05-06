def set_amount(self, amount):
        """
        Set transaction amount
        """
        if amount:
            try:
                self.IsoMessage.FieldData(4, int(amount))
            except ValueError:
                self.IsoMessage.FieldData(4, 0)
            self.rebuild()
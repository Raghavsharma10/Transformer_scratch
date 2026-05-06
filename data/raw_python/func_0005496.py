def calculate_total_amt(self, items={}):
        """Returns the total amount/cost of items in the current invoice"""
        _items = items.items() or self.items.items()
        return sum(float(x[1].total_price) for x in _items)
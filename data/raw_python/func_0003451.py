def _callbackPlaceFillOrders(self, d):
        """ This method distringuishes notifications caused by Matched orders
            from those caused by placed orders
        """
        if isinstance(d, FilledOrder):
            self.onOrderMatched(d)
        elif isinstance(d, Order):
            self.onOrderPlaced(d)
        elif isinstance(d, UpdateCallOrder):
            self.onUpdateCallOrder(d)
        else:
            pass
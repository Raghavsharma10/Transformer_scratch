def value(self):
      """Returns the positive value to subtract from the total."""
      originalPrice = self.lineItem.totalPrice
      if self.flatRate == 0:
        return originalPrice * self.percent
      return self.flatRate
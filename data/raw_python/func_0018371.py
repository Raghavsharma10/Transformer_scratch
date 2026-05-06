def value_px(self):
        """The position in pixels of the cursor"""
        step = self.w / (self.max - self.min)
        return self.x + step * (self.get() - self.min)
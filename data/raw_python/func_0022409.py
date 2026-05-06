def click_at_coordinates(self, x, y):
        """
        Click at (x,y) coordinates.
        """
        self.device.click(int(x), int(y))
def set_screen_size(self):
        """ Setup screen size and padding

        We have need 2 free lines at the top and 2 free lines at the bottom

        """
        height, width = self.getheightwidth()
        curses.resizeterm(height, width)
        self.pad_x = 0
        self.max_y, self.max_x = (height-1, width-1)
        self.pad_h = height-3
        self.pad_w = width-2*self.pad_x
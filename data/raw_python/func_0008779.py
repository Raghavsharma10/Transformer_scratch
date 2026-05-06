def step(self, x, y):
        """
        Move from the current location to the next

        Parameters
        ----------
        x, y : int
            The current location
        """
        up_left = self.solid(x - 1, y - 1)
        up_right = self.solid(x, y - 1)
        down_left = self.solid(x - 1, y)
        down_right = self.solid(x, y)

        state = 0
        self.prev = self.next
        # which cells are filled?
        if up_left:
            state |= 1
        if up_right:
            state |= 2
        if down_left:
            state |= 4
        if down_right:
            state |= 8

        # what is the next step?
        if state in [1, 5, 13]:
            self.next = self.UP
        elif state in [2, 3, 7]:
            self.next = self.RIGHT
        elif state in [4, 12, 14]:
            self.next = self.LEFT
        elif state in [8, 10, 11]:
            self.next = self.DOWN
        elif state == 6:
            if self.prev == self.UP:
                self.next = self.LEFT
            else:
                self.next = self.RIGHT
        elif state == 9:
            if self.prev == self.RIGHT:
                self.next = self.UP
            else:
                self.next = self.DOWN
        else:
            self.next = self.NOWHERE
        return
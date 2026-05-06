def size(self, time):
        """
        Gets the size of the object at a given time.

        Args:
            time: Time value being queried.

        Returns:
            size of the object in pixels
        """
        if self.start_time <= time <= self.end_time:
            return self.masks[time - self.start_time].sum()
        else:
            return 0
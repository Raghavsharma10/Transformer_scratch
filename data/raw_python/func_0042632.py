def right(self, num=None):
        """
        WITH SLICES BEING FLAT, WE NEED A SIMPLE WAY TO SLICE FROM THE RIGHT [-num:]
        """
        if num == None:
            return self.last.node
        if num <= 0:
            return []

        if not self.list:
            self._build_list()
        return self.list[-num:]
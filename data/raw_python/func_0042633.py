def not_right(self, num):
        """
        WITH SLICES BEING FLAT, WE NEED A SIMPLE WAY TO SLICE FROM THE LEFT [:-num:]
        """
        if not self.list:
            self._build_list()

        if num == None:
            return self.list[:-1:]
        if num <= 0:
            return []

        return self.list[:-num:]
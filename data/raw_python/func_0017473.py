def device_at_index(self, index):
        """
        Returns the device at the specified index
        """
        if index >= len(self.__xid_cons):
            raise ValueError("Invalid device index")

        return self.__xid_cons[index]
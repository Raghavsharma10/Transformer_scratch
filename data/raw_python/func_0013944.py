def same_disks(self, count=2):
        """ filter self to the required number of disks with same size and type

        Select the disks with the same type and same size.  If not
        enough disks available, set self to empty.

        :param count: number of disks to retrieve
        :return: disk list
        """
        ret = self
        if len(self) > 0:
            type_counter = Counter(self.drive_type)
            drive_type, counts = type_counter.most_common()[0]
            self.set_drive_type(drive_type)

        if len(self) > 0:
            size_counter = Counter(self.capacity)
            size, counts = size_counter.most_common()[0]
            self.set_capacity(size)

        if len(self) >= count:
            indices = self.index[:count]
            self.set_indices(indices)
        else:
            self.set_indices('N/A')
        return ret
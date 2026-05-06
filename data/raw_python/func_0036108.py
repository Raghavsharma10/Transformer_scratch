def remove(self, value, count=1):
        """Remove occurences of ``value`` from the list.

        :keyword count: Number of matching values to remove.
            Default is to remove a single value.

        """
        count = self.client.lrem(self.name, value, num=count)
        if not count:
            raise ValueError("%s not in list" % value)
        return count
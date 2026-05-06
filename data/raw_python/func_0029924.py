def partition(self):
        """Convenience function for accessing the first partition in the
        partitions list, when there is only one."""

        if not self.partitions:
            return None

        if len(self.partitions) > 1:
            raise ValueError(
                "Can't use this method when there is more than one partition")

        return list(self.partitions.values())[0]
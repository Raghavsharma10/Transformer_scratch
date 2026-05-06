def sub_path(self):
        """The path of the partition source, excluding the bundle path parts.

        Includes the revision.

        """

        try:
            return os.path.join(*(self._local_parts()))
        except TypeError as e:
            raise TypeError(
                "Path failed for partition {} : {}".format(
                    self.name,
                    e.message))
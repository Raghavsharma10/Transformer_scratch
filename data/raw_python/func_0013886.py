def expand(self, new_size):
        """ expand the LUN to a new size

        :param new_size: new size in bytes.
        :return: the old size
        """
        ret = self.size_total
        resp = self.modify(size=new_size)
        resp.raise_if_err()
        return ret
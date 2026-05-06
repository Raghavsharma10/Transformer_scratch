def nonull_dict(self):
        """Like dict, but does not hold any null values.

        :return:

        """
        return {k: v for k, v in six.iteritems(self.dict) if v and k != '_codes'}
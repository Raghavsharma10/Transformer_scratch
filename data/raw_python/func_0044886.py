def _get_attrs(self, names):
        """
        Convenience function to extract multiple attributes at once

        :param names:   string of names separated by comma or space
        :return:
        """
        assert isinstance(names, str)
        names = names.replace(",", " ").split(" ")
        res = []
        for n in names:
            if n == "":
                continue
            if n not in self.__dict__:
                raise KeyError("Unknown name for Container attribute: '{}'".format(n))
            res.append(getattr(self, n))
        return res
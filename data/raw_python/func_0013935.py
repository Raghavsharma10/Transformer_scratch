def sum_sp_values(self):
        """
        return system level values (spa + spb)

        input:
        "values": {
            "spa": 385,
            "spb": 505
        },

        return:
        "values": {
            "0": 890
        },
        """
        if self.values is None:
            ret = IdValues()
        else:
            ret = IdValues({'0': sum(int(x) for x in self.values.values())})
        return ret
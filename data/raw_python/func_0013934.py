def sp_sum_values(self):
        """
        return sp level values

        input:
        "values": {
            "spa": {
                "19": "385",
                "18": "0",
                "20": "0",
                "17": "0",
                "16": "0"
            },
            "spb": {
                "19": "101",
                "18": "101",
                "20": "101",
                "17": "101",
                "16": "101"
            }
        },

        return:
        "values": {
            "spa": 385,
            "spb": 505
        },
        """
        if self.values is None:
            ret = IdValues()
        else:
            ret = IdValues({k: sum(int(x) for x in v.values()) for k, v in
                            self.values.items()})
        return ret
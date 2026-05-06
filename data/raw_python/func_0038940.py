def get_tabular_str(self):
        """Creates table-like string from fields. """
        hsp_string = ""

        try:
            hsp_list = [
                {"length": self.align_length},
                {"e-value": self.expect},
                {"score": self.score},
                {"identities": self.identities},
                {"positives": self.positives},
                {"bits": self.bits},
                {"query start": self.query_start},
                {"query end": self.query_end},
                {"subject start": self.sbjct_start},
                {"subject end": self.sbjct_end},
            ]

            for h in hsp_list:
                for k, v in h.items():
                    hsp_string += "{}\t{}\n".format(k, v)
        except:
            pass

        return hsp_string
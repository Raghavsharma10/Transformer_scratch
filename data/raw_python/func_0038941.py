def best_identities(self):
        """Returns identities of the best HSP in alignment.  """
        if len(self.hsp_list) > 0:
            return round(float(self.hsp_list[0].identities) / float(self.hsp_list[0].align_length) * 100, 1)
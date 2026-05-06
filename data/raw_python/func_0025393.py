def max_csi(self):
        """
        Calculate the maximum Critical Success Index across all probability thresholds

        Returns:
            The maximum CSI as a float
        """
        csi = self.contingency_tables["TP"] / (self.contingency_tables["TP"] + self.contingency_tables["FN"] +
                                               self.contingency_tables["FP"])
        return csi.max()
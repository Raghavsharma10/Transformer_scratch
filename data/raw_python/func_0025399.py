def brier_score_components(self):
        """
        Calculate the components of the Brier score decomposition: reliability, resolution, and uncertainty.
        """
        rel_curve = self.reliability_curve()
        total = self.frequencies["Total_Freq"].sum()
        climo_freq = float(self.frequencies["Positive_Freq"].sum()) / self.frequencies["Total_Freq"].sum()
        reliability = np.sum(self.frequencies["Total_Freq"] * (rel_curve["Bin_Start"] -
                                                               rel_curve["Positive_Relative_Freq"]) ** 2) / total
        resolution = np.sum(self.frequencies["Total_Freq"] * (rel_curve["Positive_Relative_Freq"] - climo_freq) ** 2) \
                     / total
        uncertainty = climo_freq * (1 - climo_freq)
        return reliability, resolution, uncertainty
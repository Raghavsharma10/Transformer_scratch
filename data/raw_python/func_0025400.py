def brier_score(self):
        """
        Calculate the Brier Score
        """
        reliability, resolution, uncertainty = self.brier_score_components()
        return reliability - resolution + uncertainty
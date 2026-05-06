def brier_skill_score(self):
        """
        Calculate the Brier Skill Score
        """
        reliability, resolution, uncertainty = self.brier_score_components()
        return (resolution - reliability) / uncertainty
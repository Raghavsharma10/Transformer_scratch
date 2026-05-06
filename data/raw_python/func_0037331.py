def direct_evidences(self):
        """
        :return: All available direct evidences for gene disease correlations
        :rtype: list
        """
        q = self.session.query(distinct(models.GeneDisease.direct_evidence))

        return q.all()
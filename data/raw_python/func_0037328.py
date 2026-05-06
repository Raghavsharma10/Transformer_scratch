def interaction_actions(self):
        """
        :return: List of strings for allowed interaction/actions combinations
        :rtype: list[str]
        """
        r = self.session.query(distinct(models.ChemGeneIxnInteractionAction.interaction_action)).all()
        return [x[0] for x in r]
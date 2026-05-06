def _checkIdEquality(self, requestedEffect, effect):
        """
        Tests whether a requested effect and an effect
        present in an annotation are equal.
        """
        return self._idPresent(requestedEffect) and (
            effect.term_id == requestedEffect.term_id)
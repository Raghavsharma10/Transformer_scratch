def equivalent_to(self, token):
        """
        Gets all tokens which match the character and scopes of a reference token
        :param token: :class:`esi.models.Token`
        :return: :class:`esi.managers.TokenQueryset`
        """
        return self.filter(character_id=token.character_id).require_scopes_exact(token.scopes.all()).filter(
            models.Q(user=token.user) | models.Q(user__isnull=True)).exclude(pk=token.pk)
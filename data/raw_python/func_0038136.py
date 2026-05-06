def require_scopes(self, scope_string):
        """
        :param scope_string: The required scopes.
        :type scope_string: Union[str, list]
        :return: The tokens with all requested scopes.
        :rtype: :class:`esi.managers.TokenQueryset`
        """
        scopes = _process_scopes(scope_string)
        if not scopes:
            # asking for tokens with no scopes
            return self.filter(scopes__isnull=True)
        from .models import Scope
        scope_pks = Scope.objects.filter(name__in=scopes).values_list('pk', flat=True)
        if not len(scopes) == len(scope_pks):
            # there's a scope we don't recognize, so we can't have any tokens for it
            return self.none()
        tokens = self.all()
        for pk in scope_pks:
            tokens = tokens.filter(scopes__pk=pk)
        return tokens
def require_scopes_exact(self, scope_string):
        """
        :param scope_string: The required scopes.
        :type scope_string: Union[str, list]
        :return: The tokens with only the requested scopes.
        :rtype: :class:`esi.managers.TokenQueryset`
        """
        num_scopes = len(_process_scopes(scope_string))
        pks = [v['pk'] for v in self.annotate(models.Count('scopes')).require_scopes(scope_string).filter(
            scopes__count=num_scopes).values('pk', 'scopes__id')]
        return self.filter(pk__in=pks)
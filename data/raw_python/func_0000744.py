def _apply_callables(self, acl, obj=None):
        """ Iterate over ACEs from :acl: and apply callable principals
        if any.

        Principals are passed 3 arguments on call:
            :ace: Single ACE object that looks like (action, callable,
                permission or [permission])
            :request: Current request object
            :obj: Object instance to be accessed via the ACL
        Principals must return a single ACE or a list of ACEs.

        :param acl: Sequence of valid Pyramid ACEs which will be processed
        :param obj: Object to be accessed via the ACL
        """
        new_acl = []
        for i, ace in enumerate(acl):
            principal = ace[1]
            if six.callable(principal):
                ace = principal(ace=ace, request=self.request, obj=obj)
                if not ace:
                    continue
                if not isinstance(ace[0], (list, tuple)):
                    ace = [ace]
                ace = [(a, b, validate_permissions(c)) for a, b, c in ace]
            else:
                ace = [ace]
            new_acl += ace
        return tuple(new_acl)
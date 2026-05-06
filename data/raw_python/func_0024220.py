def _apply_role_tree(self, perm_tree, role):
        """In permission tree, sets `'checked': True` for the permissions that the role has."""
        role_permissions = role.get_permissions()
        for perm in role_permissions:
            self._traverse_tree(perm_tree, perm)['checked'] = True
        return perm_tree
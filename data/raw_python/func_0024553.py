def get_roles(self):
        """
        Returns:
            Role instances according to task definition.
        """
        if self.role.exist:
            # return explicitly selected role
            return [self.role]
        else:
            roles = []
            if self.role_query_code:
                #  use given "role_query_code"
                roles = RoleModel.objects.filter(**self.role_query_code)
            elif self.unit.exist:
                # get roles from selected unit or sub-units of it
                if self.recursive_units:
                    # this returns a list, we're converting it to a Role generator!
                    roles = (RoleModel.objects.get(k) for k in
                             UnitModel.get_role_keys(self.unit.key))
                else:
                    roles = RoleModel.objects.filter(unit=self.unit)
            elif self.get_roles_from:
                # get roles from selected predefined "get_roles_from" method
                return ROLE_GETTER_METHODS[self.get_roles_from](RoleModel)

        if self.abstract_role.exist and roles:
            # apply abstract_role filtering on roles we got
            if isinstance(roles, (list, types.GeneratorType)):
                roles = [a for a in roles if a.abstract_role.key == self.abstract_role.key]
            else:
                roles = roles.filter(abstract_role=self.abstract_role)
        else:
            roles = RoleModel.objects.filter(abstract_role=self.abstract_role)

        return roles
def build_role(self, name, policies=False):
        """
        Generate role for IAM cloudformation template
        :param name: Name of role
        :param policies: List of policies to attach to this role (False = none)
        :return: Ref to new role
        """
        # Build role template
        if policies:
            role = self.__template.add_resource(
                Role(
                    self.name_strip(name),
                    AssumeRolePolicyDocument=Policy(
                        Version=self.VERSION_IAM,
                        Statement=[
                            Statement(
                                Effect=Allow,
                                Principal=Principal(
                                    "Service", self.__role_principals
                                ),
                                Action=[AssumeRole],
                            )
                        ]
                    ),
                    Path=self.__role_path,
                    ManagedPolicyArns=policies,
                ))
            # Add role to list for default policy
            self.__roles_list.append(troposphere.Ref(role))
        else:
            role = self.__template.add_resource(
                Role(
                    self.name_strip(name),
                    AssumeRolePolicyDocument=Policy(
                        Version=self.VERSION_IAM,
                        Statement=[
                            Statement(
                                Effect=Allow,
                                Principal=Principal(
                                    "Service", self.__role_principals
                                ),
                                Action=[AssumeRole],
                            )
                        ]
                    ),
                    Path=self.__role_path,
                ))
            # Add role to list for default policy
            self.__roles_list.append(troposphere.Ref(role))

        return role
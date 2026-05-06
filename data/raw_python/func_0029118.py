def build_policy(self, name, statements, roles, is_managed_policy=False):
        """
        Generate policy for IAM cloudformation template
        :param name: Name of the policy
        :param statements: The "rules" the policy should have
        :param roles: The roles associated with this policy
        :param is_managed_policy: True if managed policy
        :return: Ref to new policy
        """
        if is_managed_policy:
            policy = ManagedPolicy(
                self.name_strip(name, True),
                PolicyDocument={
                    "Version": self.VERSION_IAM,
                    "Statement": statements,
                },
                Roles=roles,
                Path=self.__role_path,
            )
        else:
            policy = PolicyType(
                self.name_strip(name, True),
                PolicyName=self.name_strip(name, True),
                PolicyDocument={
                    "Version": self.VERSION_IAM,
                    "Statement": statements,
                },
                Roles=roles,
            )

        self.__template.add_resource(policy)
        return policy
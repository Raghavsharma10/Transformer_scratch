def create_with_user(self, user_params, **kwargs):
        """
        Creates an affiliate and corresponding affiliate user
        :param user_params: kwargs for user creation
        :param kwargs:
        :return: affiliate instance
        """
        affiliate = self.create(**kwargs)
        self.api.affiliate_users.create(affiliate_id=affiliate.id, **user_params)
        return affiliate
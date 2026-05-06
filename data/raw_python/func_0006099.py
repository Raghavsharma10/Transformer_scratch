def create_user(self, myshopify_domain, password=None):
        """
        Creates and saves a ShopUser with the given domain and password.
        """
        if not myshopify_domain:
            raise ValueError('ShopUsers must have a myshopify domain')

        user = self.model(myshopify_domain=myshopify_domain)

        # Never want to be able to log on externally.
        # Authentication will be taken care of by Shopify OAuth.
        user.set_unusable_password()
        user.save(using=self._db)
        return user
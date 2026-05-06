def extra_create_kwargs(self):
        """
        Inject the domain of the current user in the new model instances.
        """
        user = self.get_agnocomplete_context()
        if user:
            _, domain = user.email.split('@')
            return {
                'domain': domain
            }
        return {}
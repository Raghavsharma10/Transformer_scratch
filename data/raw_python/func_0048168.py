def save_repository(self, repository_form, *args, **kwargs):
        """Pass through to provider RepositoryAdminSession.update_repository"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.update_bin
        if repository_form.is_for_update():
            return self.update_repository(repository_form, *args, **kwargs)
        else:
            return self.create_repository(repository_form, *args, **kwargs)
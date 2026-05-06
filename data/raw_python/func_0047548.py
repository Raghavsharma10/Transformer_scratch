def save_gradebook(self, gradebook_form, *args, **kwargs):
        """Pass through to provider GradebookAdminSession.update_gradebook"""
        # Implemented from kitosid template for -
        # osid.resource.BinAdminSession.update_bin
        if gradebook_form.is_for_update():
            return self.update_gradebook(gradebook_form, *args, **kwargs)
        else:
            return self.create_gradebook(gradebook_form, *args, **kwargs)
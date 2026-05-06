def save_gradebook_column(self, gradebook_column_form, *args, **kwargs):
        """Pass through to provider GradebookColumnAdminSession.update_gradebook_column"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if gradebook_column_form.is_for_update():
            return self.update_gradebook_column(gradebook_column_form, *args, **kwargs)
        else:
            return self.create_gradebook_column(gradebook_column_form, *args, **kwargs)
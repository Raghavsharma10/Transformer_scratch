def save_grade_entry(self, grade_entry_form, *args, **kwargs):
        """Pass through to provider GradeEntryAdminSession.update_grade_entry"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if grade_entry_form.is_for_update():
            return self.update_grade_entry(grade_entry_form, *args, **kwargs)
        else:
            return self.create_grade_entry(grade_entry_form, *args, **kwargs)
def save_grade_system(self, grade_system_form, *args, **kwargs):
        """Pass through to provider GradeSystemAdminSession.update_grade_system"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if grade_system_form.is_for_update():
            return self.update_grade_system(grade_system_form, *args, **kwargs)
        else:
            return self.create_grade_system(grade_system_form, *args, **kwargs)
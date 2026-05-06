def save_assessment(self, assessment_form, *args, **kwargs):
        """Pass through to provider AssessmentAdminSession.update_assessment"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if assessment_form.is_for_update():
            return self.update_assessment(assessment_form, *args, **kwargs)
        else:
            return self.create_assessment(assessment_form, *args, **kwargs)
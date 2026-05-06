def save_assessment_taken(self, assessment_taken_form, *args, **kwargs):
        """Pass through to provider AssessmentTakenAdminSession.update_assessment_taken"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if assessment_taken_form.is_for_update():
            return self.update_assessment_taken(assessment_taken_form, *args, **kwargs)
        else:
            return self.create_assessment_taken(assessment_taken_form, *args, **kwargs)
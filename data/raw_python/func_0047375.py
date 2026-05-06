def save_assessment_offered(self, assessment_offered_form, *args, **kwargs):
        """Pass through to provider AssessmentOfferedAdminSession.update_assessment_offered"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if assessment_offered_form.is_for_update():
            return self.update_assessment_offered(assessment_offered_form, *args, **kwargs)
        else:
            return self.create_assessment_offered(assessment_offered_form, *args, **kwargs)
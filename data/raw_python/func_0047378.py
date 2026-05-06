def get_assessment_taken_form(self, *args, **kwargs):
        """Pass through to provider AssessmentTakenAdminSession.get_assessment_taken_form_for_update"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.get_resource_form_for_update
        # This method might be a bit sketchy. Time will tell.
        if isinstance(args[-1], list) or 'assessment_taken_record_types' in kwargs:
            return self.get_assessment_taken_form_for_create(*args, **kwargs)
        else:
            return self.get_assessment_taken_form_for_update(*args, **kwargs)
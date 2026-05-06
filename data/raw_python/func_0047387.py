def save_assessment_part(self, assessment_part_form, *args, **kwargs):
        """Pass through to provider AssessmentPartAdminSession.update_assessment_part"""
        # Implemented from kitosid template for -
        # osid.resource.ResourceAdminSession.update_resource
        if assessment_part_form.is_for_update():
            return self.update_assessment_part(assessment_part_form, *args, **kwargs)
        else:
            return self.create_assessment_part(assessment_part_form, *args, **kwargs)
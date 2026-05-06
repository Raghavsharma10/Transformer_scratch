def get_assessment_part_form(self, *args, **kwargs):
        """Pass through to provider AssessmentPartAdminSession.get_assessment_part_form_for_update"""
        # This method might be a bit sketchy. Time will tell.
        if isinstance(args[-1], list) or 'assessment_part_record_types' in kwargs:
            return self.get_assessment_part_form_for_create(*args, **kwargs)
        else:
            return self.get_assessment_part_form_for_update(*args, **kwargs)
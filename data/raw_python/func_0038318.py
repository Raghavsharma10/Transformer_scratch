def create_module(self, course_id, module_name, module_position=None, module_prerequisite_module_ids=None, module_publish_final_grade=None, module_require_sequential_progress=None, module_unlock_at=None):
        """
        Create a module.

        Create and return a new module
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - module[name]
        """The name of the module"""
        data["module[name]"] = module_name

        # OPTIONAL - module[unlock_at]
        """The date the module will unlock"""
        if module_unlock_at is not None:
            data["module[unlock_at]"] = module_unlock_at

        # OPTIONAL - module[position]
        """The position of this module in the course (1-based)"""
        if module_position is not None:
            data["module[position]"] = module_position

        # OPTIONAL - module[require_sequential_progress]
        """Whether module items must be unlocked in order"""
        if module_require_sequential_progress is not None:
            data["module[require_sequential_progress]"] = module_require_sequential_progress

        # OPTIONAL - module[prerequisite_module_ids]
        """IDs of Modules that must be completed before this one is unlocked.
        Prerequisite modules must precede this module (i.e. have a lower position
        value), otherwise they will be ignored"""
        if module_prerequisite_module_ids is not None:
            data["module[prerequisite_module_ids]"] = module_prerequisite_module_ids

        # OPTIONAL - module[publish_final_grade]
        """Whether to publish the student's final grade for the course upon
        completion of this module."""
        if module_publish_final_grade is not None:
            data["module[publish_final_grade]"] = module_publish_final_grade

        self.logger.debug("POST /api/v1/courses/{course_id}/modules with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/modules".format(**path), data=data, params=params, single_item=True)
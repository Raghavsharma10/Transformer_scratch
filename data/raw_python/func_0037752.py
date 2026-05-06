def create_assignment_group(self, course_id, group_weight=None, integration_data=None, name=None, position=None, rules=None, sis_source_id=None):
        """
        Create an Assignment Group.

        Create a new assignment group for this course.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - name
        """The assignment group's name"""
        if name is not None:
            data["name"] = name

        # OPTIONAL - position
        """The position of this assignment group in relation to the other assignment groups"""
        if position is not None:
            data["position"] = position

        # OPTIONAL - group_weight
        """The percent of the total grade that this assignment group represents"""
        if group_weight is not None:
            data["group_weight"] = group_weight

        # OPTIONAL - sis_source_id
        """The sis source id of the Assignment Group"""
        if sis_source_id is not None:
            data["sis_source_id"] = sis_source_id

        # OPTIONAL - integration_data
        """The integration data of the Assignment Group"""
        if integration_data is not None:
            data["integration_data"] = integration_data

        # OPTIONAL - rules
        """The grading rules that are applied within this assignment group
        See the Assignment Group object definition for format"""
        if rules is not None:
            data["rules"] = rules

        self.logger.debug("POST /api/v1/courses/{course_id}/assignment_groups with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/assignment_groups".format(**path), data=data, params=params, single_item=True)
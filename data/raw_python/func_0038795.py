def create_assignment(self, course_id, assignment_name, assignment_allowed_extensions=None, assignment_assignment_group_id=None, assignment_assignment_overrides=None, assignment_automatic_peer_reviews=None, assignment_description=None, assignment_due_at=None, assignment_external_tool_tag_attributes=None, assignment_grade_group_students_individually=None, assignment_grading_standard_id=None, assignment_grading_type=None, assignment_group_category_id=None, assignment_integration_data=None, assignment_integration_id=None, assignment_lock_at=None, assignment_muted=None, assignment_notify_of_update=None, assignment_omit_from_final_grade=None, assignment_only_visible_to_overrides=None, assignment_peer_reviews=None, assignment_points_possible=None, assignment_position=None, assignment_published=None, assignment_submission_types=None, assignment_turnitin_enabled=None, assignment_turnitin_settings=None, assignment_unlock_at=None, assignment_vericite_enabled=None):
        """
        Create an assignment.

        Create a new assignment for this course. The assignment is created in the
        active state.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - assignment[name]
        """The assignment name."""
        data["assignment[name]"] = assignment_name

        # OPTIONAL - assignment[position]
        """The position of this assignment in the group when displaying
        assignment lists."""
        if assignment_position is not None:
            data["assignment[position]"] = assignment_position

        # OPTIONAL - assignment[submission_types]
        """List of supported submission types for the assignment.
        Unless the assignment is allowing online submissions, the array should
        only have one element.
        
        If not allowing online submissions, your options are:
          "online_quiz"
          "none"
          "on_paper"
          "online_quiz"
          "discussion_topic"
          "external_tool"
        
        If you are allowing online submissions, you can have one or many
        allowed submission types:
        
          "online_upload"
          "online_text_entry"
          "online_url"
          "media_recording" (Only valid when the Kaltura plugin is enabled)"""
        if assignment_submission_types is not None:
            self._validate_enum(assignment_submission_types, ["online_quiz", "none", "on_paper", "online_quiz", "discussion_topic", "external_tool", "online_upload", "online_text_entry", "online_url", "media_recording"])
            data["assignment[submission_types]"] = assignment_submission_types

        # OPTIONAL - assignment[allowed_extensions]
        """Allowed extensions if submission_types includes "online_upload"
        
        Example:
          allowed_extensions: ["docx","ppt"]"""
        if assignment_allowed_extensions is not None:
            data["assignment[allowed_extensions]"] = assignment_allowed_extensions

        # OPTIONAL - assignment[turnitin_enabled]
        """Only applies when the Turnitin plugin is enabled for a course and
        the submission_types array includes "online_upload".
        Toggles Turnitin submissions for the assignment.
        Will be ignored if Turnitin is not available for the course."""
        if assignment_turnitin_enabled is not None:
            data["assignment[turnitin_enabled]"] = assignment_turnitin_enabled

        # OPTIONAL - assignment[vericite_enabled]
        """Only applies when the VeriCite plugin is enabled for a course and
        the submission_types array includes "online_upload".
        Toggles VeriCite submissions for the assignment.
        Will be ignored if VeriCite is not available for the course."""
        if assignment_vericite_enabled is not None:
            data["assignment[vericite_enabled]"] = assignment_vericite_enabled

        # OPTIONAL - assignment[turnitin_settings]
        """Settings to send along to turnitin. See Assignment object definition for
        format."""
        if assignment_turnitin_settings is not None:
            data["assignment[turnitin_settings]"] = assignment_turnitin_settings

        # OPTIONAL - assignment[integration_data]
        """Data related to third party integrations, JSON string required."""
        if assignment_integration_data is not None:
            data["assignment[integration_data]"] = assignment_integration_data

        # OPTIONAL - assignment[integration_id]
        """Unique ID from third party integrations"""
        if assignment_integration_id is not None:
            data["assignment[integration_id]"] = assignment_integration_id

        # OPTIONAL - assignment[peer_reviews]
        """If submission_types does not include external_tool,discussion_topic,
        online_quiz, or on_paper, determines whether or not peer reviews
        will be turned on for the assignment."""
        if assignment_peer_reviews is not None:
            data["assignment[peer_reviews]"] = assignment_peer_reviews

        # OPTIONAL - assignment[automatic_peer_reviews]
        """Whether peer reviews will be assigned automatically by Canvas or if
        teachers must manually assign peer reviews. Does not apply if peer reviews
        are not enabled."""
        if assignment_automatic_peer_reviews is not None:
            data["assignment[automatic_peer_reviews]"] = assignment_automatic_peer_reviews

        # OPTIONAL - assignment[notify_of_update]
        """If true, Canvas will send a notification to students in the class
        notifying them that the content has changed."""
        if assignment_notify_of_update is not None:
            data["assignment[notify_of_update]"] = assignment_notify_of_update

        # OPTIONAL - assignment[group_category_id]
        """If present, the assignment will become a group assignment assigned
        to the group."""
        if assignment_group_category_id is not None:
            data["assignment[group_category_id]"] = assignment_group_category_id

        # OPTIONAL - assignment[grade_group_students_individually]
        """If this is a group assignment, teachers have the options to grade
        students individually. If false, Canvas will apply the assignment's
        score to each member of the group. If true, the teacher can manually
        assign scores to each member of the group."""
        if assignment_grade_group_students_individually is not None:
            data["assignment[grade_group_students_individually]"] = assignment_grade_group_students_individually

        # OPTIONAL - assignment[external_tool_tag_attributes]
        """Hash of external tool parameters if submission_types is ["external_tool"].
        See Assignment object definition for format."""
        if assignment_external_tool_tag_attributes is not None:
            data["assignment[external_tool_tag_attributes]"] = assignment_external_tool_tag_attributes

        # OPTIONAL - assignment[points_possible]
        """The maximum points possible on the assignment."""
        if assignment_points_possible is not None:
            data["assignment[points_possible]"] = assignment_points_possible

        # OPTIONAL - assignment[grading_type]
        """The strategy used for grading the assignment.
        The assignment defaults to "points" if this field is omitted."""
        if assignment_grading_type is not None:
            self._validate_enum(assignment_grading_type, ["pass_fail", "percent", "letter_grade", "gpa_scale", "points"])
            data["assignment[grading_type]"] = assignment_grading_type

        # OPTIONAL - assignment[due_at]
        """The day/time the assignment is due.
        Accepts times in ISO 8601 format, e.g. 2014-10-21T18:48:00Z."""
        if assignment_due_at is not None:
            data["assignment[due_at]"] = assignment_due_at

        # OPTIONAL - assignment[lock_at]
        """The day/time the assignment is locked after.
        Accepts times in ISO 8601 format, e.g. 2014-10-21T18:48:00Z."""
        if assignment_lock_at is not None:
            data["assignment[lock_at]"] = assignment_lock_at

        # OPTIONAL - assignment[unlock_at]
        """The day/time the assignment is unlocked.
        Accepts times in ISO 8601 format, e.g. 2014-10-21T18:48:00Z."""
        if assignment_unlock_at is not None:
            data["assignment[unlock_at]"] = assignment_unlock_at

        # OPTIONAL - assignment[description]
        """The assignment's description, supports HTML."""
        if assignment_description is not None:
            data["assignment[description]"] = assignment_description

        # OPTIONAL - assignment[assignment_group_id]
        """The assignment group id to put the assignment in.
        Defaults to the top assignment group in the course."""
        if assignment_assignment_group_id is not None:
            data["assignment[assignment_group_id]"] = assignment_assignment_group_id

        # OPTIONAL - assignment[muted]
        """Whether this assignment is muted.
        A muted assignment does not send change notifications
        and hides grades from students.
        Defaults to false."""
        if assignment_muted is not None:
            data["assignment[muted]"] = assignment_muted

        # OPTIONAL - assignment[assignment_overrides]
        """List of overrides for the assignment.
        NOTE: The assignment overrides feature is in beta."""
        if assignment_assignment_overrides is not None:
            data["assignment[assignment_overrides]"] = assignment_assignment_overrides

        # OPTIONAL - assignment[only_visible_to_overrides]
        """Whether this assignment is only visible to overrides
        (Only useful if 'differentiated assignments' account setting is on)"""
        if assignment_only_visible_to_overrides is not None:
            data["assignment[only_visible_to_overrides]"] = assignment_only_visible_to_overrides

        # OPTIONAL - assignment[published]
        """Whether this assignment is published.
        (Only useful if 'draft state' account setting is on)
        Unpublished assignments are not visible to students."""
        if assignment_published is not None:
            data["assignment[published]"] = assignment_published

        # OPTIONAL - assignment[grading_standard_id]
        """The grading standard id to set for the course.  If no value is provided for this argument the current grading_standard will be un-set from this course.
        This will update the grading_type for the course to 'letter_grade' unless it is already 'gpa_scale'."""
        if assignment_grading_standard_id is not None:
            data["assignment[grading_standard_id]"] = assignment_grading_standard_id

        # OPTIONAL - assignment[omit_from_final_grade]
        """Whether this assignment is counted towards a student's final grade."""
        if assignment_omit_from_final_grade is not None:
            data["assignment[omit_from_final_grade]"] = assignment_omit_from_final_grade

        self.logger.debug("POST /api/v1/courses/{course_id}/assignments with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/assignments".format(**path), data=data, params=params, single_item=True)
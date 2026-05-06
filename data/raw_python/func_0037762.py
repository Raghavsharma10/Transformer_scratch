def update_course_settings(self, course_id, allow_student_discussion_editing=None, allow_student_discussion_topics=None, allow_student_forum_attachments=None, allow_student_organized_groups=None, hide_distribution_graphs=None, hide_final_grades=None, home_page_announcement_limit=None, lock_all_announcements=None, restrict_student_future_view=None, restrict_student_past_view=None, show_announcements_on_home_page=None):
        """
        Update course settings.

        Can update the following course settings:
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - allow_student_discussion_topics
        """Let students create discussion topics"""
        if allow_student_discussion_topics is not None:
            data["allow_student_discussion_topics"] = allow_student_discussion_topics

        # OPTIONAL - allow_student_forum_attachments
        """Let students attach files to discussions"""
        if allow_student_forum_attachments is not None:
            data["allow_student_forum_attachments"] = allow_student_forum_attachments

        # OPTIONAL - allow_student_discussion_editing
        """Let students edit or delete their own discussion posts"""
        if allow_student_discussion_editing is not None:
            data["allow_student_discussion_editing"] = allow_student_discussion_editing

        # OPTIONAL - allow_student_organized_groups
        """Let students organize their own groups"""
        if allow_student_organized_groups is not None:
            data["allow_student_organized_groups"] = allow_student_organized_groups

        # OPTIONAL - hide_final_grades
        """Hide totals in student grades summary"""
        if hide_final_grades is not None:
            data["hide_final_grades"] = hide_final_grades

        # OPTIONAL - hide_distribution_graphs
        """Hide grade distribution graphs from students"""
        if hide_distribution_graphs is not None:
            data["hide_distribution_graphs"] = hide_distribution_graphs

        # OPTIONAL - lock_all_announcements
        """Disable comments on announcements"""
        if lock_all_announcements is not None:
            data["lock_all_announcements"] = lock_all_announcements

        # OPTIONAL - restrict_student_past_view
        """Restrict students from viewing courses after end date"""
        if restrict_student_past_view is not None:
            data["restrict_student_past_view"] = restrict_student_past_view

        # OPTIONAL - restrict_student_future_view
        """Restrict students from viewing courses before start date"""
        if restrict_student_future_view is not None:
            data["restrict_student_future_view"] = restrict_student_future_view

        # OPTIONAL - show_announcements_on_home_page
        """Show the most recent announcements on the Course home page (if a Wiki, defaults to five announcements, configurable via home_page_announcement_limit)"""
        if show_announcements_on_home_page is not None:
            data["show_announcements_on_home_page"] = show_announcements_on_home_page

        # OPTIONAL - home_page_announcement_limit
        """Limit the number of announcements on the home page if enabled via show_announcements_on_home_page"""
        if home_page_announcement_limit is not None:
            data["home_page_announcement_limit"] = home_page_announcement_limit

        self.logger.debug("PUT /api/v1/courses/{course_id}/settings with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/settings".format(**path), data=data, params=params, no_data=True)
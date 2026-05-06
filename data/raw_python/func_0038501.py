def create_new_discussion_topic_courses(self, course_id, allow_rating=None, assignment=None, attachment=None, delayed_post_at=None, discussion_type=None, group_category_id=None, is_announcement=None, lock_at=None, message=None, only_graders_can_rate=None, pinned=None, podcast_enabled=None, podcast_has_student_posts=None, position_after=None, published=None, require_initial_post=None, sort_by_rating=None, title=None):
        """
        Create a new discussion topic.

        Create an new discussion topic for the course or group.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - title
        """no description"""
        if title is not None:
            data["title"] = title

        # OPTIONAL - message
        """no description"""
        if message is not None:
            data["message"] = message

        # OPTIONAL - discussion_type
        """The type of discussion. Defaults to side_comment if not value is given. Accepted values are 'side_comment', for discussions that only allow one level of nested comments, and 'threaded' for fully threaded discussions."""
        if discussion_type is not None:
            self._validate_enum(discussion_type, ["side_comment", "threaded"])
            data["discussion_type"] = discussion_type

        # OPTIONAL - published
        """Whether this topic is published (true) or draft state (false). Only
        teachers and TAs have the ability to create draft state topics."""
        if published is not None:
            data["published"] = published

        # OPTIONAL - delayed_post_at
        """If a timestamp is given, the topic will not be published until that time."""
        if delayed_post_at is not None:
            data["delayed_post_at"] = delayed_post_at

        # OPTIONAL - lock_at
        """If a timestamp is given, the topic will be scheduled to lock at the
        provided timestamp. If the timestamp is in the past, the topic will be
        locked."""
        if lock_at is not None:
            data["lock_at"] = lock_at

        # OPTIONAL - podcast_enabled
        """If true, the topic will have an associated podcast feed."""
        if podcast_enabled is not None:
            data["podcast_enabled"] = podcast_enabled

        # OPTIONAL - podcast_has_student_posts
        """If true, the podcast will include posts from students as well. Implies
        podcast_enabled."""
        if podcast_has_student_posts is not None:
            data["podcast_has_student_posts"] = podcast_has_student_posts

        # OPTIONAL - require_initial_post
        """If true then a user may not respond to other replies until that user has
        made an initial reply. Defaults to false."""
        if require_initial_post is not None:
            data["require_initial_post"] = require_initial_post

        # OPTIONAL - assignment
        """To create an assignment discussion, pass the assignment parameters as a
        sub-object. See the {api:AssignmentsApiController#create Create an Assignment API}
        for the available parameters. The name parameter will be ignored, as it's
        taken from the discussion title. If you want to make a discussion that was
        an assignment NOT an assignment, pass set_assignment = false as part of
        the assignment object"""
        if assignment is not None:
            data["assignment"] = assignment

        # OPTIONAL - is_announcement
        """If true, this topic is an announcement. It will appear in the
        announcement's section rather than the discussions section. This requires
        announcment-posting permissions."""
        if is_announcement is not None:
            data["is_announcement"] = is_announcement

        # OPTIONAL - pinned
        """If true, this topic will be listed in the "Pinned Discussion" section"""
        if pinned is not None:
            data["pinned"] = pinned

        # OPTIONAL - position_after
        """By default, discussions are sorted chronologically by creation date, you
        can pass the id of another topic to have this one show up after the other
        when they are listed."""
        if position_after is not None:
            data["position_after"] = position_after

        # OPTIONAL - group_category_id
        """If present, the topic will become a group discussion assigned
        to the group."""
        if group_category_id is not None:
            data["group_category_id"] = group_category_id

        # OPTIONAL - allow_rating
        """If true, users will be allowed to rate entries."""
        if allow_rating is not None:
            data["allow_rating"] = allow_rating

        # OPTIONAL - only_graders_can_rate
        """If true, only graders will be allowed to rate entries."""
        if only_graders_can_rate is not None:
            data["only_graders_can_rate"] = only_graders_can_rate

        # OPTIONAL - sort_by_rating
        """If true, entries will be sorted by rating."""
        if sort_by_rating is not None:
            data["sort_by_rating"] = sort_by_rating

        # OPTIONAL - attachment
        """A multipart/form-data form-field-style attachment.
        Attachments larger than 1 kilobyte are subject to quota restrictions."""
        if attachment is not None:
            data["attachment"] = attachment

        self.logger.debug("POST /api/v1/courses/{course_id}/discussion_topics with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/discussion_topics".format(**path), data=data, params=params, no_data=True)
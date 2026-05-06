def update_create_front_page_courses(self, course_id, wiki_page_body=None, wiki_page_editing_roles=None, wiki_page_notify_of_update=None, wiki_page_published=None, wiki_page_title=None):
        """
        Update/create front page.

        Update the title or contents of the front page
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - wiki_page[title]
        """The title for the new page. NOTE: changing a page's title will change its
        url. The updated url will be returned in the result."""
        if wiki_page_title is not None:
            data["wiki_page[title]"] = wiki_page_title

        # OPTIONAL - wiki_page[body]
        """The content for the new page."""
        if wiki_page_body is not None:
            data["wiki_page[body]"] = wiki_page_body

        # OPTIONAL - wiki_page[editing_roles]
        """Which user roles are allowed to edit this page. Any combination
        of these roles is allowed (separated by commas).
        
        "teachers":: Allows editing by teachers in the course.
        "students":: Allows editing by students in the course.
        "members":: For group wikis, allows editing by members of the group.
        "public":: Allows editing by any user."""
        if wiki_page_editing_roles is not None:
            self._validate_enum(wiki_page_editing_roles, ["teachers", "students", "members", "public"])
            data["wiki_page[editing_roles]"] = wiki_page_editing_roles

        # OPTIONAL - wiki_page[notify_of_update]
        """Whether participants should be notified when this page changes."""
        if wiki_page_notify_of_update is not None:
            data["wiki_page[notify_of_update]"] = wiki_page_notify_of_update

        # OPTIONAL - wiki_page[published]
        """Whether the page is published (true) or draft state (false)."""
        if wiki_page_published is not None:
            data["wiki_page[published]"] = wiki_page_published

        self.logger.debug("PUT /api/v1/courses/{course_id}/front_page with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/front_page".format(**path), data=data, params=params, single_item=True)
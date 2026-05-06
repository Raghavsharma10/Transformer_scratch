def get_courses(self):
        """
        use the base_url and auth data from the configuration to list all courses the user is subscribed to
        """
        log.info("Listing Courses...")
        courses = json.loads(self._get('/api/courses').text)["courses"]
        courses = [Course.from_response(course) for course in courses]
        log.debug("Courses: %s" % [str(entry) for entry in courses])
        return courses
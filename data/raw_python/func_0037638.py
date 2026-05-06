def reset_course_favorites(self):
        """
        Reset course favorites.

        Reset the current user's course favorites to the default
        automatically generated list of enrolled courses
        """
        path = {}
        data = {}
        params = {}

        self.logger.debug("DELETE /api/v1/users/self/favorites/courses with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/users/self/favorites/courses".format(**path), data=data, params=params, no_data=True)
def export_data(self, export_to, delete_content=False):
        """
        Export course data to another course.
        Does not include any user data.

        :param bool delete_content: (optional) Delete content \
            from source course.

        Example Usage::

        >>> import muddle
        >>> muddle.course(10).export_data(12)
        """
        params = {'wsfunction': 'core_course_import_course',
                  'importfrom': self.course_id,
                  'importto': export_to,
                  'deletecontent': int(delete_content)}
        params.update(self.request_params)

        return requests.post(self.api_url, params=params, verify=False)
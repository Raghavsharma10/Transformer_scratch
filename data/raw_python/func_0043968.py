def delete(self):
        """
        Deletes a specified courses

        Example Usage::

        >>> import muddle
        >>> muddle.course(10).delete()
        """

        params = {'wsfunction': 'core_course_delete_courses',
                  'courseids[0]': self.course_id}
        params.update(self.request_params)

        return requests.post(self.api_url, params=params, verify=False)
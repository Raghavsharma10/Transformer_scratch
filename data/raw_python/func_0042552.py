def get_project(self, project_id):
        """ Get project info """
        try:
            result = self._request('/getproject/',
                                   {'projectid': project_id})
            return TildaProject(**result)
        except NetworkError:
            return []
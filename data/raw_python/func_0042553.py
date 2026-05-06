def get_project_export(self, project_id):
        """ Get project info for export """
        try:
            result = self._request('/getprojectexport/',
                                   {'projectid': project_id})
            return TildaProject(**result)
        except NetworkError:
            return []
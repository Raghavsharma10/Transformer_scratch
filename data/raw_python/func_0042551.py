def get_projects_list(self):
        """ Get projects list """
        try:
            result = self._request('/getprojectslist/')
            return [TildaProject(**p) for p in result]
        except NetworkError:
            return []
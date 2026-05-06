def create_application(self, team_id, name, url=None):
        """
        Creates an application under a given team.
        :param team_id: Team identifier.
        :param name: The name of the new application being created.
        :param url: The url of where the application is located.
        """
        params = {'name': name}
        if url:
            params['url'] = url
        return self._request('POST', 'rest/teams/' + str(team_id) + '/applications/new', params)
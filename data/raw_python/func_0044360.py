def set_application_parameters(self, application_id, framework_type, repository_url):
        """
        Sets parameters for the Hybrid Analysis Mapping ThreadFix functionality.
        :param application_id: Application identifier.
        :param framework_type: The web framework the app was built on. ('None', 'DETECT', 'JSP', 'SPRING_MVC')
        :param repository_url: The git repository where the source code for the application can be found.
        """
        params = {
            'frameworkType': framework_type,
            'repositoryUrl': repository_url
        }
        return self._request('POST', 'rest/applications/' + str(application_id) + '/setParameters', params)
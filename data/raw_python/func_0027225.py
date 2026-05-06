def list(self, request, *args, **kwargs):
        """
        To get a list of service settings, run **GET** against */api/service-settings/* as an authenticated user.
        Only settings owned by this user or shared settings will be listed.

        Supported filters are:

        - ?name=<text> - partial matching used for searching
        - ?type=<type> - choices: OpenStack, DigitalOcean, Amazon, JIRA, GitLab, Oracle
        - ?state=<state> - choices: New, Creation Scheduled, Creating, Sync Scheduled, Syncing, In Sync, Erred
        - ?shared=<bool> - allows to filter shared service settings
        """
        return super(ServiceSettingsViewSet, self).list(request, *args, **kwargs)
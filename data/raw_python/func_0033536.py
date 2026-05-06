def notify_deployment(self, application_id=None, application_name=None, description=None, revision=None, changelog=None, user=None):
        """
        Notify NewRelic of a deployment.
        http://newrelic.github.io/newrelic_api/NewRelicApi/Deployment.html

        :param description:
        :param revision:
        :param changelog:
        :param user:
        :return: A dictionary containing all of the returned keys from the API
        """

        endpoint = "https://rpm.newrelic.com"
        uri = "{endpoint}/deployments.xml".format(endpoint=endpoint)

        deploy_event = {}

        if not application_id is None:
            deploy_event['deployment[application_id]'] = application_id
        elif not application_name is None:
            deploy_event['deployment[app_name]'] = application_name
        else:
            raise NewRelicInvalidParameterException("Must specify either application_id or application_name.")

        if not description is None:
            deploy_event['deployment[description]'] = description

        if not revision is None:
            deploy_event['deployment[revision]'] = revision

        if not changelog is None:
            deploy_event['deployment[changelog]'] = changelog

        if not user is None:
            deploy_event['deployment[user]'] = user

        response = self._make_post_request(uri, deploy_event)
        result = {}

        for value in response:
            result[value.tag] = value.text

        return result
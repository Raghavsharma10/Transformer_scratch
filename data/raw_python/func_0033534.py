def view_applications(self):
        """
        Requires: account ID (taken from Client object)
        Returns: a list of Application objects
        Endpoint: rpm.newrelic.com
        Errors: 403 Invalid API Key
        Method: Get
        """
        endpoint = "https://rpm.newrelic.com"
        uri = "{endpoint}/accounts/{id}/applications.xml".format(endpoint=endpoint, id=self.account_id)
        response = self._make_get_request(uri)
        applications = []

        for application in response.findall('.//application'):
            application_properties = {}
            for field in application:
                application_properties[field.tag] = field.text
            applications.append(Application(application_properties))
        return applications
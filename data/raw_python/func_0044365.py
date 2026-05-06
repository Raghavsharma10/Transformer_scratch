def get_waf_rules_by_application(self, waf_id, application_id):
        """
        Returns the WAF rule text for one or all of the applications in a WAF. If the application id is -1, it will get
        rules for all apps. If the application is a valid application id, rules will be generated for that application.
        :param waf_id: WAF identifier.
        :param application_id: Application identifier.
        """
        return self._request('GET', 'rest/wafs/' + str(waf_id) + '/rules/app/' + str(application_id))
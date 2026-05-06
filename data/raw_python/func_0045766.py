def set_http_request(self, http_request):
        """Support the HTTPRequest ProxyConditionRecordType and checks for special effective agent ids"""
        self._http_request = http_request
        if 'HTTP_LTI_USER_ID' in http_request.META:
            try:
                authority = http_request.META['HTTP_LTI_TOOL_CONSUMER_INSTANCE_GUID']
            except (AttributeError, KeyError):
                authority = 'unknown_lti_consumer_instance'
            self.set_effective_agent_id(Id(
                authority=authority,
                namespace='agent.Agent',
                identifier=http_request.META['HTTP_LTI_USER_ID']))
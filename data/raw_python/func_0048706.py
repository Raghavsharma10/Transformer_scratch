def get_agent(self):
        """Gets the ``Agent`` who created this entry.

        return: (osid.authentication.Agent) - the ``Agent``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.Resource.get_avatar_template
        if not bool(self._my_map['agentId']):
            raise errors.IllegalState('this LogEntry has no agent')
        mgr = self._get_provider_manager('AUTHENTICATION')
        if not mgr.supports_agent_lookup():
            raise errors.OperationFailed('Authentication does not support Agent lookup')
        lookup_session = mgr.get_agent_lookup_session(proxy=getattr(self, "_proxy", None))
        lookup_session.use_federated_agency_view()
        osid_object = lookup_session.get_agent(self.get_agent_id())
        return osid_object
def get_commenting_agent(self):
        """Gets the agent who created this comment.

        return: (osid.authentication.Agent) - the ``Agent``
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        if not self.has_commentor():
            raise errors.IllegalState('this Comment has no commenting_agent')
        try:
            from ..authentication import managers
        except ImportError:
            raise errors.OperationFailed('failed to import authentication.managers')
        try:
            mgr = managers.AuthenticationManager()
        except:
            raise errors.OperationFailed('failed to instantiate AuthenticationManager')
        if not mgr.supports_agent_lookup():
            raise errors.OperationFailed('Authentication does not support Agent lookup')
        try:
            osid_object = mgr.get_agent_lookup_session().get_agent(self.get_commenting_agent_id())
        except:
            raise errors.OperationFailed()
        else:
            return osid_object
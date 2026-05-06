def get_agent_id(self):
        """Gets the ``Id`` of the ``Agent`` identified in this authentication credential.

        return: (osid.id.Id) - the ``Agent Id``
        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: The Agent should be determined at the
        time this credential is created.

        """
        if self._django_user is not None:
            return Id(identifier=self._django_user.get_username(),
                      namespace='osid.agent.Agent',
                      authority='MIT-OEIT')
        else:
            return Id(identifier='MC3GUE$T@MIT.EDU',
                      namespace='osid.agent.Agent',
                      authority='MIT-OEIT')
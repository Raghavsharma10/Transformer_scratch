def get_agent_id(self):
        """Gets the ``Id`` of the ``Agent`` identified in this authentication credential.

        :return: the ``Agent Id``
        :rtype: ``osid.id.Id``

        *compliance: mandatory -- This method must be implemented.*
        *implementation notes*: The Agent should be determined at the
        time this credential is created.

        """
        if self._django_user is not None:
            if self._use_user_id:
                identifier = self._django_user.id
            else:
                identifier = self._django_user.get_username()
            return Id(identifier=identifier,
                      namespace='osid.agent.Agent',
                      authority='MIT-ODL')
        else:
            # perhaps this id should come from django settings?
            return Id(identifier='MC3GUE$T@MIT.EDU',
                      namespace='osid.agent.Agent',
                      authority='MIT-ODL')
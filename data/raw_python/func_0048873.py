def get_agent(self):
        """Gets the ``Agent`` identified in this authentication credential.

        :return: the ``Agent``
        :rtype: ``osid.authentication.Agent``
        :raise: ``OperationFailed`` -- unable to complete request

        *compliance: mandatory -- This method must be implemented.*

        """
        agent_id = self.get_agent_id()
        return Agent(identifier=agent_id.identifier,
                     namespace=agent_id.namespace,
                     authority=agent_id.authority)
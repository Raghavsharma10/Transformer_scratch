def get_agents_by_resource(self, resource_id):
        """Gets the list of ``Agents`` mapped to a ``Resource``.

        arg:    resource_id (osid.id.Id): ``Id`` of a ``Resource``
        return: (osid.authentication.AgentList) - list of agents
        raise:  NotFound - ``resource_id`` is not found
        raise:  NullArgument - ``resource_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        agent_list = []
        for agent_id in self.get_agent_ids_by_resource(resource_id):
            agent_list.append(Agent(agent_id))
        return AgentList(agent_list)
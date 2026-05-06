def commit(self, revision, is_amend=False):
        """
        :param revision:
        :type revision: :class:`revision.data.Revision`
        :param is_amend:
        :type is_amend: boolean
        :return: The Orchestrator instance (for method chaining)
        :rtype: :class:`revision.orchestrator.Orchestrator`
        """
        if not isinstance(revision, Revision):
            raise InvalidArgType()

        if not self.current_client:
            raise ClientNotSpecified()

        if is_amend:
            self.current_client.save(revision)
        else:
            self.current_client.write()
            self.current_client.save(revision)

        return self
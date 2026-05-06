def from_tasks(cls, tasks):
        """Construct a list of MarathonEndpoints from a list of tasks.

        :param list[:class:`marathon.models.MarathonTask`] tasks: list of tasks to parse

        :rtype: list[:class:`MarathonEndpoint`]
        """

        endpoints = [
            [
                MarathonEndpoint(task.app_id, task.service_ports[
                                 port_index], task.host, task.id, port)
                for port_index, port in enumerate(task.ports)
            ]
            for task in tasks
        ]
        # Flatten result
        return [item for sublist in endpoints for item in sublist]
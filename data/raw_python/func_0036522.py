def get_active_nodes(self):
        """
        Returns a dictionary of lists, where the key is the name of a service
        and the list includes all active nodes associated with that service.
        """
        # the -1 4 -1 args are the filters <proxy_id> <type> <server_id>,
        # -1 for all proxies, 4 for servers only, -1 for all servers
        stats_response = self.send_command("show stat -1 4 -1")
        if not stats_response:
            return []

        lines = stats_response.split("\n")
        fields = lines.pop(0).split(",")
        # the first field is the service name, which we key off of so
        # it's not included in individual node records
        fields.pop(0)

        active_nodes = collections.defaultdict(list)

        for line in lines:
            values = line.split(",")
            service_name = values.pop(0)
            active_nodes[service_name].append(
                dict(
                    (fields[i], values[i])
                    for i in range(len(fields))
                )
            )

        return active_nodes
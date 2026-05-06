def fetch_routing_info(self, address):
        """ Fetch raw routing info from a given router address.

        :param address: router address
        :return: list of routing records or
                 None if no connection could be established
        :raise ServiceUnavailable: if the server does not support routing or
                                   if routing support is broken
        """
        metadata = {}
        records = []

        def fail(md):
            if md.get("code") == "Neo.ClientError.Procedure.ProcedureNotFound":
                raise RoutingProtocolError("Server {!r} does not support routing".format(address))
            else:
                raise RoutingProtocolError("Routing support broken on server {!r}".format(address))

        try:
            with self.acquire_direct(address) as cx:
                _, _, server_version = (cx.server.agent or "").partition("/")
                # TODO 2.0: remove old routing procedure
                if server_version and Version.parse(server_version) >= Version((3, 2)):
                    log_debug("[#%04X]  C: <ROUTING> query=%r", cx.local_port, self.routing_context or {})
                    cx.run("CALL dbms.cluster.routing.getRoutingTable({context})",
                           {"context": self.routing_context}, on_success=metadata.update, on_failure=fail)
                else:
                    log_debug("[#%04X]  C: <ROUTING> query={}", cx.local_port)
                    cx.run("CALL dbms.cluster.routing.getServers", {}, on_success=metadata.update, on_failure=fail)
                cx.pull_all(on_success=metadata.update, on_records=records.extend)
                cx.sync()
                routing_info = [dict(zip(metadata.get("fields", ()), values)) for values in records]
                log_debug("[#%04X]  S: <ROUTING> info=%r", cx.local_port, routing_info)
            return routing_info
        except RoutingProtocolError as error:
            raise ServiceUnavailable(*error.args)
        except ServiceUnavailable:
            self.deactivate(address)
            return None
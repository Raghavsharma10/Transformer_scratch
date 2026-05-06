def report_up(self, service, port):
        """
        Report the given service's present node as up by creating/updating
        its respective znode in Zookeeper and setting the znode's data to
        the serialized representation of the node.

        Waits for zookeeper to be connected before taking any action.
        """
        wait_on_any(self.connected, self.shutdown)

        node = Node.current(service, port)

        path = self.path_of(service, node)
        data = node.serialize().encode()

        znode = self.client.exists(path)

        if not znode:
            logger.debug("ZNode at %s does not exist, creating new one.", path)
            self.client.create(path, value=data, ephemeral=True, makepath=True)
        elif znode.owner_session_id != self.client.client_id[0]:
            logger.debug("ZNode at %s not owned by us, recreating.", path)
            txn = self.client.transaction()
            txn.delete(path)
            txn.create(path, value=data, ephemeral=True)
            txn.commit()
        else:
            logger.debug("Setting node value to %r", data)
            self.client.set(path, data)
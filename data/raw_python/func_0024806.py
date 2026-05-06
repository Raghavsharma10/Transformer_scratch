async def rename(self, name):
        """Change name of node."""
        set_node_name = SetNodeName(pyvlx=self.pyvlx, node_id=self.node_id, name=name)
        await set_node_name.do_api_call()
        if not set_node_name.success:
            raise PyVLXException("Unable to rename node")
        self.name = name
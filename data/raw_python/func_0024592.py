async def _load_node(self, node_id):
        """Load single node via API."""
        get_node_information = GetNodeInformation(pyvlx=self.pyvlx, node_id=node_id)
        await get_node_information.do_api_call()
        if not get_node_information.success:
            raise PyVLXException("Unable to retrieve node information")
        notification_frame = get_node_information.notification_frame
        node = convert_frame_to_node(self.pyvlx, notification_frame)
        if node is not None:
            self.add(node)
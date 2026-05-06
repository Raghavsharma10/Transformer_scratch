async def _load_all_nodes(self):
        """Load all nodes via API."""
        get_all_nodes_information = GetAllNodesInformation(pyvlx=self.pyvlx)
        await get_all_nodes_information.do_api_call()
        if not get_all_nodes_information.success:
            raise PyVLXException("Unable to retrieve node information")
        self.clear()
        for notification_frame in get_all_nodes_information.notification_frames:
            node = convert_frame_to_node(self.pyvlx, notification_frame)
            if node is not None:
                self.add(node)
async def process_frame(self, frame):
        """Update nodes via frame, usually received by house monitor."""
        if isinstance(frame, FrameNodeStatePositionChangedNotification):
            if frame.node_id not in self.pyvlx.nodes:
                return
            node = self.pyvlx.nodes[frame.node_id]
            if isinstance(node, OpeningDevice):
                node.position = Position(frame.current_position)
                await node.after_update()
        elif isinstance(frame, FrameGetAllNodesInformationNotification):
            if frame.node_id not in self.pyvlx.nodes:
                return
            node = self.pyvlx.nodes[frame.node_id]
            if isinstance(node, OpeningDevice):
                node.position = Position(frame.current_position)
                await node.after_update()
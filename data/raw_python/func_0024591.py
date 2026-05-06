async def load(self, node_id=None):
        """Load nodes from KLF 200, if no node_id is specified all nodes are loaded."""
        if node_id is not None:
            await self._load_node(node_id=node_id)
        else:
            await self._load_all_nodes()
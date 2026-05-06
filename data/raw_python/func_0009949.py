def LayerTree_loadSnapshot(self, tiles):
		"""
		Function path: LayerTree.loadSnapshot
			Domain: LayerTree
			Method name: loadSnapshot
		
			Parameters:
				Required arguments:
					'tiles' (type: array) -> An array of tiles composing the snapshot.
			Returns:
				'snapshotId' (type: SnapshotId) -> The id of the snapshot.
		
			Description: Returns the snapshot identifier.
		"""
		assert isinstance(tiles, (list, tuple)
		    ), "Argument 'tiles' must be of type '['list', 'tuple']'. Received type: '%s'" % type(
		    tiles)
		subdom_funcs = self.synchronous_command('LayerTree.loadSnapshot', tiles=tiles
		    )
		return subdom_funcs
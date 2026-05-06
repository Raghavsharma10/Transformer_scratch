def _update_trsys(self, event):
        """Transform object(s) have changed for this Node; assign these to the
        visual's TransformSystem.
        """
        doc = self.document_node
        scene = self.scene_node
        root = self.root_node
        self.transforms.visual_transform = self.node_transform(scene)
        self.transforms.scene_transform = scene.node_transform(doc)
        self.transforms.document_transform = doc.node_transform(root)

        Node._update_trsys(self, event)
def _update_trsys(self, event):
        """Called when  has changed.
        
        This allows the node and its children to react (notably, VisualNode
        uses this to update its TransformSystem).
        
        Note that this method is only called when one transform is replaced by
        another; it is not called if an existing transform internally changes
        its state.
        """
        for ch in self.children:
            ch._update_trsys(event)
        self.events.transform_change()
        self.update()
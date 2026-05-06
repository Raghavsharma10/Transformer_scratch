def _set_clipper(self, node, clipper):
        """Assign a clipper that is inherited from a parent node.

        If *clipper* is None, then remove any clippers for *node*.
        """
        if node in self._clippers:
            self.detach(self._clippers.pop(node))
        if clipper is not None:
            self.attach(clipper)
            self._clippers[node] = clipper
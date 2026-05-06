def open_files(self, idx=0):
        """Call method |Devices.open_files| of the |Nodes| and |Elements|
        objects currently handled by the |HydPy| object."""
        self.elements.open_files(idx=idx)
        self.nodes.open_files(idx=idx)
def export(self, location):
        """Export the Bazaar repository at the url to the destination location"""
        temp_dir = tempfile.mkdtemp('-export', 'pip-')
        self.unpack(temp_dir)
        if os.path.exists(location):
            # Remove the location to make sure Bazaar can export it correctly
            rmtree(location)
        try:
            call_subprocess([self.cmd, 'export', location], cwd=temp_dir,
                            filter_stdout=self._filter, show_stdout=False)
        finally:
            rmtree(temp_dir)
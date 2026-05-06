def sync_objects_in(self):
        """Synchronize from records to objects"""
        self.dstate = self.STATES.BUILDING
        self.build_source_files.record_to_objects()
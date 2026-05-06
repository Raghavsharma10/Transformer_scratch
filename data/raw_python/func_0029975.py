def update_schema(self):
        """Propagate schema object changes to file records"""

        self.commit()
        self.build_source_files.schema.objects_to_record()
        self.commit()
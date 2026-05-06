def clean_source_files(self):
        """Remove the schema.csv and source_schema.csv files"""

        self.build_source_files.file(File.BSFILE.SOURCESCHEMA).remove()
        self.build_source_files.file(File.BSFILE.SCHEMA).remove()
        self.commit()
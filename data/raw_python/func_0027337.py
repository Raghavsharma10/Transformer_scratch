def import_lsdinst(self, struct_data):
        """import from an lsdinst struct"""
        self.name = struct_data['name']
        self.automate = struct_data['data']['automate']
        self.pan = struct_data['data']['pan']

        if self.table is not None:
            self.table.import_lsdinst(struct_data)
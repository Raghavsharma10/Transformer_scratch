def from_dict(self, description):
        """Configures the task store to be the task_store described 
            in description"""
        assert(self.ident == description['ident'])
        self.partitions = description['partitions']
        self.indices    = description['indices']
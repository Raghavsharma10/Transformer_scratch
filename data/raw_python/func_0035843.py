def create_tables(self):
        """
        create tables in database (if they don't already exist)
        """
        cdir = os.path.dirname( os.path.realpath(__file__) )
        
        # table schemas -------------------------------------
        schema = os.path.join(cdir,"data","partsmaster.sql")
        if self.debug:
            print(self.hdr,"parts master schema is ",schema)
        
        self.populate(schema)
        
        schema = os.path.join(cdir,"data","approvedmfg.sql")
        if self.debug:
            print(self.hdr,"approved mfg list schema is ",schema)
        
        self.populate(schema)
        
        schema = os.path.join(cdir,"data","attachment.sql")
        if self.debug:
            print(self.hdr,"attachment schema is ",schema)
        
        self.populate(schema)

        schema = os.path.join(cdir,"data","bom.sql")
        if self.debug:
            print(self.hdr,"bill of materials schema is ",schema)
        
        self.populate(schema)
        
        return
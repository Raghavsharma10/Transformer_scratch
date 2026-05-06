def _init_table_elementTree(self, xml=True, db_table=True):
        """Create a table 
        """
        # <table> tag object
        t_table =  xtree.Element('table')
        # <table xmlns='' securityLevel='' htpps=''>
        if not self.table_attr :
            self.table_attr = self._TAB_ATTR
        for attr in self.table_attr.items() :
            t_table.set(*attr)

        # <meta>
        t_meta = xtree.SubElement(t_table, 'meta')

        # Loop over a sorted key,value of class attributes while ignoring table_attr and name
        for key, value in [(k,v) for k,v in sorted(self.__dict__.items(), key=lambda x: x[0]) if k not in ('table_attr','name') ]:
            if isinstance(value, list): # Works for element like sampleQuery
                for elt in value:
                    t_tag = xtree.SubElement(t_meta, key) # setting attribute name as a tag name
                    t_tag.text = elt # Setting attribute  value as text
            else:
                t_tag = xtree.SubElement(t_meta,key)
                t_tag.text = value
      
        ## <bindings>
        t_bindings = xtree.SubElement(t_table, 'bindings')
        ##

        self.etree = t_table
        return t_table
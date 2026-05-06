def get_for_org(self, org_id, ids, skip_cache=False):
        '''
        Returns objects for the given identifiers
        If called with a list returns a list, else returns a single entity
        '''
        t = type(ids)
        if t != list:
            ids = [ids]
            
        objs = []
        
        
        for id in ids:
            if id == '' or type(id) is not str: # Empty string
                objs.append(None)
                continue

            if skip_cache ==False:
                obj = self.get_from_cache(org_id, id)
            else:
                obj = None
                
            if obj is None:
                xml = self.request_obj(org_id, id)
                obj = self.create_obj_from_xml(id, xml)
                self.cache(obj) # Will cache either a real object, or a BioCycEntityNotFound
                
            if obj: # Found
                objs.append(obj)
            else:  # Not found (BioCycEntityNotFound)
                objs.append(None)
                
        if t != list:
            return objs[0] 
        else:
            return objs
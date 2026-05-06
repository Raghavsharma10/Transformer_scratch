def get_from_cache(self, org_id, id):
        '''
        Get an object from the cache
        
        Use all cache folders available (primary first, then secondary in order) and look for the ID in the dir
        if found unpickle and return the object, else return False
        
        FIXME: Check for expiry of object! Return false is expired (will auto-refetch and overwrite)
        '''
        current_time = datetime.now()
        
        # Check memory cache first
        if id in self.memory_cache[org_id]:
            obj = self.memory_cache[org_id][id]
            if obj.created_at > current_time - self.expire_records_after:
                return obj
        
        for cache in [self.cache_path] + self.secondary_cache_paths:
            read_path = os.path.join( cache, org_id, id )
            try:
                with open(read_path, 'rb') as f:
                    obj = pickle.load(f)

            except:
                # Continue to try the next cache
                pass 
                
            else:
                # It worked so we have obj
                # Check for expiry date; if it's not expired return it else continue
                if obj.created_at > current_time - self.expire_records_after:
                    # If we're here it mustn't be in the memory cache
                    self.memory_cache[org_id][id] = obj
                    if len(self.memory_cache[org_id]) > self.max_memory_cache:
                        self.memory_cache[org_id].popitem(last=False)

                    return obj
                    
                # Else continue looking

        # We found nothing (or all expired)
        return None
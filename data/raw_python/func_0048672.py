def cache(self, obj):
        '''
        Store an object in the cache (this allows temporarily assigning a new cache
        for exploring the DB without affecting the stored version
        '''
        # Check cache path exists for current obj
        write_path = os.path.join( self.cache_path, obj.org_id )
        if not os.path.exists( write_path ): 
            mkdir_p( write_path )

        with open(os.path.join( write_path, obj.id ), 'wb') as f:
            pickle.dump( obj, f )
        
        # Add to localstore (keep track of numbers of objects, etc.)
        self.add_to_localstore(obj)   
        self.add_to_names(obj)
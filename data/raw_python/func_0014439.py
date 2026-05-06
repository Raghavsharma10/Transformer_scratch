def upload_collection_configs(self, collection, fs_path):
        '''
        Uploads collection configurations from a specified directory to zookeeper. 
        
        '''
        coll_path = fs_path
        if not os.path.isdir(coll_path):
            raise ValueError("{} Doesn't Exist".format(coll_path))
        self._upload_dir(coll_path, '/configs/{}'.format(collection))
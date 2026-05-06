def download_collection_configs(self, collection, fs_path):
        '''
        Downloads ZK Directory to the FileSystem.

        :param collection str: Name of the collection (zk config name)
        :param fs_path str: Destination filesystem path. 
        '''
        
        if not self.kz.exists('/configs/{}'.format(collection)):
            raise ZookeeperError("Collection doesn't exist in Zookeeper. Current Collections are: {} ".format(self.kz.get_children('/configs')))
        self._download_dir('/configs/{}'.format(collection), fs_path + os.sep + collection)
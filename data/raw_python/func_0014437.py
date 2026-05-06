def copy_config(self, original, new):
        '''
        Copies collection configs into a new folder. Can be used to create new collections based on existing configs. 

        Basically, copies all nodes under /configs/original to /configs/new.

        :param original str: ZK name of original config
        :param new str: New name of the ZK config. 
        '''
        if not self.kz.exists('/configs/{}'.format(original)):
            raise ZookeeperError("Collection doesn't exist in Zookeeper. Current Collections are: {}".format(self.kz.get_children('/configs')))
        base = '/configs/{}'.format(original)
        nbase = '/configs/{}'.format(new)
        self._copy_dir(base, nbase)
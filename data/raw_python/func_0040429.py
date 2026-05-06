def unregister(self, itemtype, item_id):
        '''
        deregister the item in zookeeper /list/
        itemtype must be a Zooborg constant
        item_id must be a string
        '''
        if itemtype not in [ZooConst.CLIENT, ZooConst.WORKER, ZooConst.BROKER]:
            raise Exception('Zooborg.unregister: invalid type')
        self.initconn()
        self.zk.ensure_path("/distark/" + itemtype + "/list")
        path=''.join(['/distark/' + itemtype + '/list/', item_id])
        if self.zk.exists(path):
            self.zk.delete(path, recursive=True)
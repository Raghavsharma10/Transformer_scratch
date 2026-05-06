def getList(self, listtype):
        '''
        listtype must be a Zooborg constant
        '''
        if listtype not in [ZooConst.CLIENT, ZooConst.WORKER, ZooConst.BROKER]:
            raise Exception('Zooborg.getList: invalid type')
        self.initconn()
        return self.zk.get_children('/distark/' + listtype + '/list')
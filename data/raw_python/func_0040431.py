def getConf(self, conftype):
        '''
        conftype must be a Zooborg constant
        '''
        zooconf={}
        if conftype not in [ZooConst.CLIENT, ZooConst.WORKER, ZooConst.BROKER]:
            raise Exception('Zooborg.getConf: invalid type')

        self.initconn()
        if conftype in [ZooConst.CLIENT, ZooConst.WORKER]:
            zooconf={'broker': {'connectionstr': None}}
            zoopath='/distark/' + conftype + '/conf/broker/connectionstr'
            zooconf['broker']['connectionstr'], stat = self.zk.get(zoopath)

        if conftype in [ZooConst.BROKER]:
            zooconf={'bindstr': None}
            zoopath='/distark/' + conftype + '/conf/bindstr'
            zooconf['bindstr'], stat = self.zk.get(zoopath)

        return zooconf
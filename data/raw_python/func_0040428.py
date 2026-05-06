def getConf(self, conftype):
        '''
        conftype must be a Zooborg constant
        '''
        if conftype not in [ZooConst.CLIENT, ZooConst.WORKER, ZooConst.BROKER]:
            raise Exception('Zooborg.getConf: invalid type')

        zooconf={}

        #TODO: specialconf entries for the mock

        if conftype == ZooConst.CLIENT:
            zooconf['broker'] = {}
            zooconf['broker']['connectionstr'] = b"tcp://localhost:5555"

        elif conftype == ZooConst.BROKER:
            zooconf['bindstr']=b"tcp://*:5555"

        elif conftype == ZooConst.WORKER:
            zooconf['broker'] = {}
            zooconf['broker']['connectionstr'] = b"tcp://localhost:5555"

        else:
            raise Exception("ZooBorgconftype unknown")


        return zooconf
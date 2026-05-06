def insertBulkBlock(self, blockDump):
        """
        API to insert a bulk block

        :param blockDump: Output of the block dump command, example can be found in https://svnweb.cern.ch/trac/CMSDMWM/browser/DBS/trunk/Client/tests/dbsclient_t/unittests/blockdump.dict
        :type blockDump: dict

        """
        #We first check if the first lumi section has event_count or not
        frst = True
        if (blockDump['files'][0]['file_lumi_list'][0]).get('event_count') == None: frst = False
        # when frst == True, we are looking for event_count == None in the data, if we did not find None (redFlg = False), 
        # eveything is good. Otherwise, we have to remove all even_count in lumis and raise exception.
        # when frst == False, weare looking for event_count != None in the data, if we did not find Not None (redFlg = False),        # everything is good. Otherwise, we have to remove all even_count in lumis and raise exception.     
        redFlag = False
        if frst == True:
            eventCT = (fl.get('event_count') == None for f in  blockDump['files'] for fl in f['file_lumi_list'])
        else:                 
            eventCT = (fl.get('event_count') != None for f in  blockDump['files'] for fl in f['file_lumi_list'])

        redFlag = any(eventCT)
        if redFlag:
            for f in blockDump['files']:
                for fl in f['file_lumi_list']:
                    if 'event_count' in fl: del fl['event_count']
        result =  self.__callServer("bulkblocks", data=blockDump, callmethod='POST' )
        if redFlag:
            raise dbsClientException("Mixed event_count per lumi in the block: %s" %blockDump['block']['block_name'], 
                                    "The block was inserted into DBS, but you need to check if the data is valid.")
        else:
            return result
def colAdd(self,name="",desc="",unit="",comment="",coltype=0,data=[],pos=None):
        """
        column types:
            0: Y
            1: Disregard
            2: Y Error
            3: X
            4: Label
            5: Z
            6: X Error
        """
        if pos is None:
            pos=len(self.colNames)
        self.colNames.insert(pos,name)
        self.colDesc.insert(pos,desc)
        self.colUnits.insert(pos,unit)
        self.colComments.insert(pos,comment)
        self.colTypes.insert(pos,coltype)
        self.colData.insert(pos,data)
        return
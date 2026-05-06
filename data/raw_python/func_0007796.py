def validAspects(self, ID, aspList):
        """ Returns a list with the aspects an object 
        makes with the other six planets, considering a
        list of possible aspects. 
        
        """
        obj = self.chart.getObject(ID)
        res = []
        
        for otherID in const.LIST_SEVEN_PLANETS:
            if ID == otherID:
                continue
            
            otherObj = self.chart.getObject(otherID)
            aspType = aspects.aspectType(obj, otherObj, aspList)
            if aspType != const.NO_ASPECT:
                res.append({
                    'id': otherID,
                    'asp': aspType,
                })
        return res
def get(self, ID):
        """ Returns an object, house or angle 
        from the chart.
        
        """
        if ID.startswith('House'):
            return self.getHouse(ID)
        elif ID in const.LIST_ANGLES:
            return self.getAngle(ID)
        else:
            return self.getObject(ID)
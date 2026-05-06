def aspectsByCat(self, ID, aspList):
        """ Returns the aspects an object makes with the
        other six planets, separated by category (applicative,
        separative, exact). 
        Aspects must be within orb of the object.
        
        """
        res = {
            const.APPLICATIVE: [],
            const.SEPARATIVE: [],
            const.EXACT: [],
            const.NO_MOVEMENT: []
        }
        
        objA = self.chart.getObject(ID)
        valid = self.validAspects(ID, aspList)
        for elem in valid:
            objB = self.chart.getObject(elem['id'])
            asp = aspects.getAspect(objA, objB, aspList)
            role = asp.getRole(objA.id)
            if role['inOrb']:
                movement = role['movement']
                res[movement].append({
                    'id': objB.id,
                    'asp': asp.type,
                    'orb': asp.orb
                })

        return res
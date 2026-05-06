def eqMutualReceptions(self):
        """ Returns a list with mutual receptions with the 
        object and other planets, when the reception is the 
        same for both (both ruler or both exaltation).
        
        It basically return a list with every ruler-ruler and 
        exalt-exalt mutual receptions
        
        """
        mrs = self.reMutualReceptions()
        res = []
        for ID, receptions in mrs.items():
            for pair in receptions:
                if pair[0] == pair[1]:
                    res.append(pair[0])
        return res
def getScoreProperties(self):
        """ Returns the accidental dignity score of the object 
        as dict. 
        
        """
        obj = self.obj
        score = {}
        
        # Peregrine
        isPeregrine = essential.isPeregrine(obj.id, obj.sign, obj.signlon)
        score['peregrine'] = -5 if isPeregrine else 0
        
        # Ruler-Ruler and Exalt-Exalt mutual receptions
        mr = self.eqMutualReceptions()
        score['mr_ruler'] = +5 if 'ruler' in mr else 0
        score['mr_exalt'] = +4 if 'exalt' in mr else 0
        
        # House scores
        score['house'] = self.houseScore()
        
        # Joys
        score['joy_sign'] = +3 if self.inSignJoy() else 0
        score['joy_house'] = +2 if self.inHouseJoy() else 0
        
        # Relations with sun
        score['cazimi'] = +5 if self.isCazimi() else 0
        score['combust'] = -6 if self.isCombust() else 0
        score['under_sun'] = -4 if self.isUnderSun() else 0
        score['no_under_sun'] = 0
        if obj.id != const.SUN and not self.sunRelation():
            score['no_under_sun'] = +5
        
        # Light
        score['light'] = 0
        if obj.id != const.SUN:
            score['light'] = +1 if self.isAugmentingLight() else -1
            
        # Orientality
        score['orientality'] = 0
        if obj.id in [const.SATURN, const.JUPITER, const.MARS]:
            score['orientality'] = +2 if self.isOriental() else -2
        elif obj.id in [const.VENUS, const.MERCURY, const.MOON]:
            score['orientality'] = -2 if self.isOriental() else +2
        
        # Moon nodes
        score['north_node'] = -3 if self.isConjNorthNode() else 0
        score['south_node'] = -5 if self.isConjSouthNode() else 0
        
        # Direction and speed
        score['direction'] = 0
        if obj.id not in [const.SUN, const.MOON]:
            score['direction'] = +4 if obj.isDirect() else -5
        score['speed'] = +2 if obj.isFast() else -2
        
        # Aspects to benefics
        aspBen = self.aspectBenefics()
        score['benefic_asp0'] = +5 if const.CONJUNCTION in aspBen else 0
        score['benefic_asp120'] = +4 if const.TRINE in aspBen else 0
        score['benefic_asp60'] = +3 if const.SEXTILE in aspBen else 0
        
        # Aspects to malefics
        aspMal = self.aspectMalefics()
        score['malefic_asp0'] = -5 if const.CONJUNCTION in aspMal else 0
        score['malefic_asp180'] = -4 if const.OPPOSITION in aspMal else 0
        score['malefic_asp90'] = -3 if const.SQUARE in aspMal else 0
        
        # Auxily and Surround
        score['auxilied'] = +5 if self.isAuxilied() else 0
        score['surround'] = -5 if self.isSurrounded() else 0
        
        # Voc and Feral
        score['feral'] = -3 if self.isFeral() else 0
        score['void'] = -2 if (self.isVoc() and score['feral'] == 0) else 0
        
        # Haiz
        haiz = self.haiz()
        score['haiz'] = 0
        if haiz == HAIZ:
            score['haiz'] = +3
        elif haiz == CHAIZ:
            score['haiz'] = -2
            
        # Moon via combusta
        score['viacombusta'] = 0
        if obj.id == const.MOON and viaCombusta(obj):
            score['viacombusta'] = -2
            
        return score
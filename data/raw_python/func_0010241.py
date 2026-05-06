def swap(self, side='AB', inplace=False):
        '''
        :side: - optional string
        :inplace: - optional boolean
        :return: Triangle with flipped side.

        The optional side paramater should have one of three values:
        AB, BC, or AC.

        Changes the order of the triangle's points, swapping the
        specified points. Doing so will change the results of isCCW
        and ccw.

        '''
        try:
            flipset = {'AB': (self.B.xyz, self.A.xyz, self.C.xyz),
                       'BC': (self.A.xyz, self.C.xyz, self.B.xyz),
                       'AC': (self.C.xyz, self.B.xyz, self.A.xyz)}[side]
        except KeyError as e:
            raise KeyError(str(e))

        if inplace:
            self.ABC = flipset
            return self

        return Triangle(flipset)
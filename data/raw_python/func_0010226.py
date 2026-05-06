def containsPoint(self, point, Zorder=False):
        '''
        :param: point  - Point subclass
        :param: Zorder - optional Boolean

        Is true if the point is contain in the rectangle or
        along the rectangle's edges.

        If Zorder is True, the method will check point.z for
        equality with the rectangle origin's Z coordinate.

        '''
        if not point.isBetweenX(self.A, self.B):
            return False
        if not point.isBetweenY(self.A, self.D):
            return False

        if Zorder:
            return point.z == self.origin.z

        return True
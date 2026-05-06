def doesIntersect(self, other):
        '''
        :param: other - Line subclass
        :return: boolean

        Returns True iff:
           ccw(self.A,self.B,other.A) * ccw(self.A,self.B,other.B) <= 0
           and
           ccw(other.A,other.B,self.A) * ccw(other.A,other.B,self.B) <= 0

        '''
        if self.A.ccw(self.B, other.A) * self.A.ccw(self.B, other.B) > 0:
            return False

        if other.A.ccw(other.B, self.A) * other.A.ccw(other.B, self.B) > 0:
            return False

        return True
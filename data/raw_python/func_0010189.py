def heronsArea(self):
        '''
        Heron's forumla for computing the area of a triangle, float.

        Performance note: contains a square root.

        '''
        s = self.semiperimeter

        return math.sqrt(s * ((s - self.a) * (s - self.b) * (s - self.c)))
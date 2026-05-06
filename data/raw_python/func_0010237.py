def segments(self):
        '''
        A list of the Triangle's line segments [AB, BC, AC], list.

        '''
        return [Segment(self.AB),
                Segment(self.BC),
                Segment(self.AC)]
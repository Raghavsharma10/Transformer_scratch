def lr_padding(self, terms):
        '''
        Pad doc from the left and right before adding,
        depending on what's in self.lpad and self.rpad
        If any of them is '', then don't pad there. 
        '''
        lpad = rpad = []
        if self.lpad:
            lpad = [self.lpad] * (self.n - 1) 
        if self.rpad:
            rpad = [self.rpad] * (self.n - 1) 
        return lpad + terms + rpad
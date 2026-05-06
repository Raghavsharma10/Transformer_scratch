def _synDelayParams(self):
        '''
        set up the detailed synaptic delay parameters,
        loc is mean delay,
        scale is std with low bound cutoff,
        assumes numpy.random.normal is used later
        '''
        delays = {}
        #mean delays
        loc = np.zeros((len(self.y), len(self.X)))
        loc[:, 0] = self.delays[0]
        loc[:, 1::2] = self.delays[0]
        loc[:, 2::2] = self.delays[1]
        #standard deviations
        scale = loc * self.delay_rel_sd
        
        #prepare output
        delay_loc = {}
        for i, y in enumerate(self.y):
            delay_loc.update({y : loc[i]})
        
        delay_scale = {}
        for i, y in enumerate(self.y):
            delay_scale.update({y : scale[i]})
                
        return delay_loc, delay_scale
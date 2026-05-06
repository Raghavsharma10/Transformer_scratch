def sample_many(self, num_samples = 2000):
        """
        Generates a given number of trajectories, using the method sample(). 
        Returns a fixmat with the generated data.
        
        Parameters:
            num_samples : int, optional
                The number of trajectories that shall be generated.
        """     
        x = []
        y = []
        fix = []
        sample = []
        
        # XXX: Delete ProgressBar
        pbar = ProgressBar(widgets=[Percentage(),Bar()], maxval=num_samples).start()
        
        for s in range(0, num_samples):
            for i, (xs, ys) in enumerate(self.sample()):
                x.append(xs)
                y.append(ys)
                fix.append(i+1)
                sample.append(s)
            pbar.update(s+1)
            
        fields = {'fix':np.array(fix), 'y':np.array(y), 'x':np.array(x)}
        param = {'pixels_per_degree':self.fm.pixels_per_degree}
        out =  fixmat.VectorFixmatFactory(fields, param)
        return out
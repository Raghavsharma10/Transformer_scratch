def initializeData(self, fit = None, full_H1=None, max_length = 40,
            in_deg = True):
        """
        Prepares the data to be replicated. Calculates the second-order length 
        and angle dependencies between saccades and stores them in a fitted 
        histogram.
        
        Parameters:
            fit : function, optional
                The method to use for fitting the histogram
            full_H1 : twodimensional numpy.ndarray, optional
                Where applicable, the distribution of angle and length
                differences to replicate with dimensions [73,361]
        """
        a, l, ad, ld = anglendiff(self.fm, roll=1, return_abs = True)
        if in_deg:
            self.fm.pixels_per_degree = 1
            
        samples = np.zeros([3, len(l[0])])
        samples[0] = l[0]/self.fm.pixels_per_degree
        samples[1] = np.roll(l[0]/self.fm.pixels_per_degree,-1)
        samples[2] = np.roll(reshift(ad[0]),-1)
        z = np.any(np.isnan(samples), axis=0)
        samples = samples[:,~np.isnan(samples).any(0)]
           
        if full_H1 is None:   
            self.full_H1 = []
            for i in range(1, int(ceil(max_length+1))):
                idx = np.logical_and(samples[0]<=i, samples[0]>i-1)
                if idx.any():
                    self.full_H1.append(makeHist(samples[2][idx], samples[1][idx], fit=fit, 
                                                bins=[np.linspace(0,max_length-1,max_length),np.linspace(-180,180,361)]))
                    # Sometimes if there's only one sample present there seems to occur a problem
                    # with histogram calculation and the hist is filled with nans. In this case, dismiss
                    # the hist.
                    if np.isnan(self.full_H1[-1]).any():
                        self.full_H1[-1] = np.array([])
                    self.nosamples.append(len(samples[2][idx]))
                else:
                    self.full_H1.append(np.array([]))
                    self.nosamples.append(0)
        else:
            self.full_H1 = full_H1
                
        self.firstLenAng_cumsum, self.firstLenAng_shape = (
                                        compute_cumsum(firstSacDist(self.fm)))
        self.probability_cumsum = []
       
        for i in range(len(self.full_H1)):
            if self.full_H1[i] == []:
                self.probability_cumsum.append(np.array([]))
            else:
                self.probability_cumsum.append(np.cumsum(self.full_H1[i].flat))
               
        self.trajLen_cumsum, self.trajLen_borders = trajLenDist(self.fm)
        
        min_distance = 1/np.array([min((np.unique(self.probability_cumsum[i]) \
                        -np.roll(np.unique(self.probability_cumsum[i]),1))[1:]) \
                        for i in range(len(self.probability_cumsum))])
        # Set a minimal resolution
        min_distance[min_distance<10] = 10

        self.linind = {}
        for i in range(len(self.probability_cumsum)):
            self.linind['self.probability_cumsum '+repr(i)] = np.linspace(0,1,min_distance[i])[0:-1]
        
        for elem in [self.firstLenAng_cumsum, self.trajLen_cumsum]:
            self.linind[elem] = np.linspace(0, 1, 1/min((np.unique((elem))-np.roll(np.unique((elem)),1))[1:]))[0:-1]
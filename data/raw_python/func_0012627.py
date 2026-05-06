def make_reg_data(self, feature_list=None, all_controls=False):    
        """ 
        Generates two M x N matrices with M feature values at fixations for 
        N features. Controls are a random sample out of all non-fixated regions 
        of an image or fixations of the same subject group on a randomly chosen 
        image. Fixations are pooled over all subjects in the calling fixmat.
       
        Parameters : 
            all_controls : bool
                if True, all non-fixated points on a feature map are takes as
                control values. If False, controls are fixations from the same
                subjects but on one other randomly chosen image of the same
                category
            feature_list : list of strings
                contains names of all features that are used to generate
                the feature value matrix (--> number of dimensions in the 
                model). 
                ...note: this list has to be sorted !

        Returns : 
            N x M matrix of N control feature values per feature (M).
            Rows = Feature number /type
            Columns = Feature values
        """
        if not 'x' in self.fieldnames():
            raise RuntimeError("""make_reg_data expects to find
        (x,y) locations in self.x and self.y. But self.x does not exist""")

        on_image = (self.x >= 0) & (self.x <= self.image_size[1])
        on_image = on_image & (self.y >= 0) & (self.y <= self.image_size[0])
        assert on_image.all(), "All Fixations need to be on the image"
        assert len(np.unique(self.filenumber) > 1), "Fixmat has to have more than one filenumber"
        self.x = self.x.astype(int)
        self.y = self.y.astype(int)
        
        if feature_list == None:
            feature_list = np.sort(self._categories._features)
        all_act = np.zeros((len(feature_list), 1)) * np.nan
        all_ctrls = all_act.copy()
                            
        for (cfm, imgs) in self.by_cat():
            # make a list of all filenumbers in this category and then 
            # choose one random filenumber without replacement
            imfiles = np.array(imgs.images()) # array makes a copy of the list
            ctrl_imgs = imfiles.copy()
            np.random.shuffle(ctrl_imgs)
            while (imfiles == ctrl_imgs).any():
                np.random.shuffle(ctrl_imgs)
            for (imidx, img) in enumerate(imfiles):
                xact = cfm.x[cfm.filenumber == img]
                yact = cfm.y[cfm.filenumber == img]
                if all_controls:
                # take a sample the same length as the actuals out of every 
                # non-fixated point in the feature map
                    idx = np.ones(self.image_size)
                    idx[cfm.y[cfm.filenumber == img], 
                        cfm.x[cfm.filenumber == img]] = 0
                    yctrl, xctrl = idx.nonzero()
                    idx = np.random.randint(0, len(yctrl), len(xact))
                    yctrl = yctrl[idx]
                    xctrl = xctrl[idx]
                    del idx
                else:
                    xctrl = cfm.x[cfm.filenumber == ctrl_imgs[imidx]]
                    yctrl = cfm.y[cfm.filenumber == ctrl_imgs[imidx]]
                # initialize arrays for this filenumber
                actuals = np.zeros((1, len(xact))) * np.nan
                controls = np.zeros((1, len(xctrl))) * np.nan                
                
                for feature in feature_list:
                    # get the feature map
                    fmap = imgs[img][feature]
                    actuals = np.vstack((actuals, fmap[yact, xact]))
                    controls = np.vstack((controls, fmap[yctrl, xctrl]))
                all_act = np.hstack((all_act, actuals[1:, :]))
                all_ctrls = np.hstack((all_ctrls, controls[1:, :]))
        return (all_act[:, 1:], all_ctrls[:, 1:])
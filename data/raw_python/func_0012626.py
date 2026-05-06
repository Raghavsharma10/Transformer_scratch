def add_feature_values(self, features):
        """
        Adds feature values of feature 'feature' to all fixations in 
        the calling fixmat.
        
        For fixations out of the image boundaries, NaNs are returned.
        The function generates a new attribute field named with the
        string in features that contains an np.array listing feature
        values for every fixation in the fixmat.
        
        .. note:: The calling fixmat must have been constructed with an 
        stimuli.Categories object
        
        Parameters:
            features : string
                list of feature names for which feature values are extracted.
        """
        if not 'x' in self.fieldnames():
            raise RuntimeError("""add_feature_values expects to find
        (x,y) locations in self.x and self.y. But self.x does not exist""")
 
        if not self._categories:
            raise RuntimeError(
            '''"%s" does not exist as a fieldname and the
            fixmat does not have a Categories object (no features 
            available. The fixmat has these fields: %s''' \
            %(features, str(self._fields))) 
        for feature in features:
            # initialize new field with NaNs
            feat_vals = np.zeros([len(self.x)]) * np.nan 
            for (cat_mat, imgs) in self.by_cat():
                for img in np.unique(cat_mat.filenumber).astype(int):
                    fmap = imgs[img][feature]
                    on_image = (self.x >= 0) & (self.x <= self.image_size[1])
                    on_image = on_image & (self.y >= 0) & (self.y <= self.image_size[0])
                    idx = (self.category == imgs.category) & \
                          (self.filenumber == img) & \
                          (on_image.astype('bool'))
                    feat_vals[idx] = fmap[self.y[idx].astype('int'), 
                        self.x[idx].astype('int')]
            # setattr(self, feature, feat_vals)
            self.add_field(feature, feat_vals)
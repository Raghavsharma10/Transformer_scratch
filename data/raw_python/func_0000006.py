def set_seeds(self, seeds):
        """
        Function for manual seed setting. Sets variable seeds and prepares
        voxels for density model.
        :param seeds: ndarray (0 - nothing, 1 - object, 2 - background,
        3 - object just hard constraints, no model training, 4 - background 
        just hard constraints, no model training)
        """
        if self.img.shape != seeds.shape:
            raise Exception("Seeds must be same size as input image")

        self.seeds = seeds.astype("int8")
        self.voxels1 = self.img[self.seeds == 1]
        self.voxels2 = self.img[self.seeds == 2]
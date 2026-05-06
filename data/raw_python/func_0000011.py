def debug_get_reconstructed_similarity(
        self,
        data3d=None,
        voxelsize=None,
        seeds=None,
        area_weight=1,
        hard_constraints=True,
        return_unariesalt=False,
    ):
        """
        Use actual model to calculate similarity. If no input is given the last image is used.
        :param data3d:
        :param voxelsize:
        :param seeds:
        :param area_weight:
        :param hard_constraints:
        :param return_unariesalt:
        :return:
        """
        if data3d is None:
            data3d = self.img
        if voxelsize is None:
            voxelsize = self.voxelsize
        if seeds is None:
            seeds = self.seeds

        unariesalt = self.__create_tlinks(
            data3d,
            voxelsize,
            # voxels1, voxels2,
            seeds,
            area_weight,
            hard_constraints,
        )
        if return_unariesalt:
            return unariesalt
        else:
            return self._reshape_unariesalt_to_similarity(unariesalt, data3d.shape)
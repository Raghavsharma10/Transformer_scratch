def debug_show_reconstructed_similarity(
        self,
        data3d=None,
        voxelsize=None,
        seeds=None,
        area_weight=1,
        hard_constraints=True,
        show=True,
        bins=20,
        slice_number=None,
    ):
        """
        Show tlinks.
        :param data3d: ndarray with input data
        :param voxelsize:
        :param seeds:
        :param area_weight:
        :param hard_constraints:
        :param show:
        :param bins: histogram bins number
        :param slice_number:
        :return:
        """

        unariesalt = self.debug_get_reconstructed_similarity(
            data3d,
            voxelsize=voxelsize,
            seeds=seeds,
            area_weight=area_weight,
            hard_constraints=hard_constraints,
            return_unariesalt=True,
        )

        self._debug_show_unariesalt(
            unariesalt, show=show, bins=bins, slice_number=slice_number
        )
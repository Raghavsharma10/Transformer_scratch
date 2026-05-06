def _ssgc_prepare_data_and_run_computation(
        self,
        # voxels1, voxels2,
        hard_constraints=True,
        area_weight=1,
    ):
        """
        Setting of data.
        You need set seeds if you want use hard_constraints.
        """
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        # import pdb; pdb.set_trace() # BREAKPOINT

        unariesalt = self.__create_tlinks(
            self.img,
            self.voxelsize,
            # voxels1, voxels2,
            self.seeds,
            area_weight,
            hard_constraints,
        )
        #  některém testu  organ semgmentation dosahují unaries -15. což je podiné
        # stačí vyhodit print před if a je to vidět
        logger.debug("unaries %.3g , %.3g" % (np.max(unariesalt), np.min(unariesalt)))
        # create potts pairwise
        # pairwiseAlpha = -10
        pairwise = -(np.eye(2) - 1)
        pairwise = (self.segparams["pairwise_alpha"] * pairwise).astype(np.int32)
        # pairwise = np.array([[0,30],[30,0]]).astype(np.int32)
        # print pairwise

        self.iparams = {}

        if self.segparams["use_boundary_penalties"]:
            sigma = self.segparams["boundary_penalties_sigma"]
            # set boundary penalties function
            # Default are penalties based on intensity differences
            boundary_penalties_fcn = lambda ax: self._boundary_penalties_array(
                axis=ax, sigma=sigma
            )
        else:
            boundary_penalties_fcn = None
        nlinks = self.__create_nlinks(
            self.img, boundary_penalties_fcn=boundary_penalties_fcn
        )

        self.stats["tlinks shape"].append(unariesalt.reshape(-1, 2).shape)
        self.stats["nlinks shape"].append(nlinks.shape)
        # we flatten the unaries
        # result_graph = cut_from_graph(nlinks, unaries.reshape(-1, 2),
        # pairwise)
        start = time.time()
        if self.debug_images:
            self._debug_show_unariesalt(unariesalt)
        result_graph = pygco.cut_from_graph(nlinks, unariesalt.reshape(-1, 2), pairwise)
        elapsed = time.time() - start
        self.stats["gc time"] = elapsed
        result_labeling = result_graph.reshape(self.img.shape)

        return result_labeling
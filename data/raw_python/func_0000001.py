def __multiscale_gc_lo2hi_run(self):  # , pyed):
        """
        Run Graph-Cut segmentation with refinement of low resolution multiscale graph.
        In first step is performed normal GC on low resolution data
        Second step construct finer grid on edges of segmentation from first
        step.
        There is no option for use without `use_boundary_penalties`
        """
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()
        self._msgc_lo2hi_resize_init()
        self.__msgc_step0_init()

        hard_constraints = self.__msgc_step12_low_resolution_segmentation()
        # ===== high resolution data processing
        seg = self.__msgc_step3_discontinuity_localization()

        self.stats["t3.1"] = (time.time() - self._start_time)
        graph = Graph(
            seg,
            voxelsize=self.voxelsize,
            nsplit=self.segparams["block_size"],
            edge_weight_table=self._msgc_npenalty_table,
            compute_low_nodes_index=True,
        )

        # graph.run() = graph.generate_base_grid() + graph.split_voxels()
        # graph.run()
        graph.generate_base_grid()
        self.stats["t3.2"] = (time.time() - self._start_time)
        graph.split_voxels()

        self.stats["t3.3"] = (time.time() - self._start_time)

        self.stats.update(graph.stats)
        self.stats["t4"] = (time.time() - self._start_time)
        mul_mask, mul_val = self.__msgc_tlinks_area_weight_from_low_segmentation(seg)
        area_weight = 1
        unariesalt = self.__create_tlinks(
            self.img,
            self.voxelsize,
            self.seeds,
            area_weight=area_weight,
            hard_constraints=hard_constraints,
            mul_mask=None,
            mul_val=None,
        )
        # N-links prepared
        self.stats["t5"] = (time.time() - self._start_time)
        un, ind = np.unique(graph.msinds, return_index=True)
        self.stats["t6"] = (time.time() - self._start_time)

        self.stats["t7"] = (time.time() - self._start_time)
        unariesalt2_lo2hi = np.hstack(
            [unariesalt[ind, 0, 0].reshape(-1, 1), unariesalt[ind, 0, 1].reshape(-1, 1)]
        )
        nlinks_lo2hi = np.hstack([graph.edges, graph.edges_weights.reshape(-1, 1)])
        if self.debug_images:
            import sed3

            ed = sed3.sed3(unariesalt[:, :, 0].reshape(self.img.shape))
            ed.show()
            import sed3

            ed = sed3.sed3(unariesalt[:, :, 1].reshape(self.img.shape))
            ed.show()
            # ed = sed3.sed3(seg)
            # ed.show()
            # import sed3
            # ed = sed3.sed3(graph.data)
            # ed.show()
            # import sed3
            # ed = sed3.sed3(graph.msinds)
            # ed.show()

        # nlinks, unariesalt2, msinds = self.__msgc_step45678_construct_graph(area_weight, hard_constraints, seg)
        # self.__msgc_step9_finish_perform_gc_and_reshape(nlinks, unariesalt2, msinds)
        self.__msgc_step9_finish_perform_gc_and_reshape(
            nlinks_lo2hi, unariesalt2_lo2hi, graph.msinds
        )
        self._msgc_lo2hi_resize_clean_finish()
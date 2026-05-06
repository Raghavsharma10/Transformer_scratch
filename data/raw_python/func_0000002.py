def __multiscale_gc_hi2lo_run(self):  # , pyed):
        """
        Run Graph-Cut segmentation with simplifiyng of high resolution multiscale graph.
        In first step is performed normal GC on low resolution data
        Second step construct finer grid on edges of segmentation from first
        step.
        There is no option for use without `use_boundary_penalties`
        """
        # from PyQt4.QtCore import pyqtRemoveInputHook
        # pyqtRemoveInputHook()

        self.__msgc_step0_init()
        hard_constraints = self.__msgc_step12_low_resolution_segmentation()
        # ===== high resolution data processing
        seg = self.__msgc_step3_discontinuity_localization()
        nlinks, unariesalt2, msinds = self.__msgc_step45678_hi2lo_construct_graph(
            hard_constraints, seg
        )
        self.__msgc_step9_finish_perform_gc_and_reshape(nlinks, unariesalt2, msinds)
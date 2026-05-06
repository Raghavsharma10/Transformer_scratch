def run(self, run_fit_model=True):
        """
        Run the Graph Cut segmentation according to preset parameters.

        :param run_fit_model: Allow to skip model fit when the model is prepared before
        :return:
        """

        if run_fit_model:
            self.fit_model(self.img, self.voxelsize, self.seeds)

        self._start_time = time.time()
        if self.segparams["method"].lower() in ("graphcut", "gc"):
            self.__single_scale_gc_run()
        elif self.segparams["method"].lower() in (
            "multiscale_graphcut",
            "multiscale_gc",
            "msgc",
            "msgc_lo2hi",
            "lo2hi",
            "multiscale_graphcut_lo2hi",
        ):
            logger.debug("performing multiscale Graph-Cut lo2hi")
            self.__multiscale_gc_lo2hi_run()
        elif self.segparams["method"].lower() in (
            "msgc_hi2lo",
            "hi2lo",
            "multiscale_graphcut_hi2lo",
        ):
            logger.debug("performing multiscale Graph-Cut hi2lo")
            self.__multiscale_gc_hi2lo_run()
        else:
            logger.error("Unknown segmentation method: " + self.segparams["method"])
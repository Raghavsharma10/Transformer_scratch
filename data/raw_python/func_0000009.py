def __similarity_for_tlinks_obj_bgr(
        self,
        data,
        voxelsize,
        # voxels1, voxels2,
        # seeds, otherfeatures=None
    ):
        """
        Compute edge values for graph cut tlinks based on image intensity
        and texture.
        """
        # self.fit_model(data, voxelsize, seeds)
        # There is a need to have small vaues for good fit
        # R(obj) = -ln( Pr (Ip | O) )
        # R(bck) = -ln( Pr (Ip | B) )
        # Boykov2001b
        # ln is computed in likelihood
        tdata1 = (-(self.mdl.likelihood_from_image(data, voxelsize, 1))) * 10
        tdata2 = (-(self.mdl.likelihood_from_image(data, voxelsize, 2))) * 10

        # to spare some memory
        dtype = np.int16
        if np.any(tdata1 > 32760):
            dtype = np.float32
        if np.any(tdata2 > 32760):
            dtype = np.float32

        if self.segparams["use_apriori_if_available"] and self.apriori is not None:
            logger.debug("using apriori information")
            gamma = self.segparams["apriori_gamma"]
            a1 = (-np.log(self.apriori * 0.998 + 0.001)) * 10
            a2 = (-np.log(0.999 - (self.apriori * 0.998))) * 10
            # logger.debug('max ' + str(np.max(tdata1)) + ' min ' + str(np.min(tdata1)))
            # logger.debug('max ' + str(np.max(tdata2)) + ' min ' + str(np.min(tdata2)))
            # logger.debug('max ' + str(np.max(a1)) + ' min ' + str(np.min(a1)))
            # logger.debug('max ' + str(np.max(a2)) + ' min ' + str(np.min(a2)))
            tdata1u = (((1 - gamma) * tdata1) + (gamma * a1)).astype(dtype)
            tdata2u = (((1 - gamma) * tdata2) + (gamma * a2)).astype(dtype)
            tdata1 = tdata1u
            tdata2 = tdata2u
            # logger.debug('   max ' + str(np.max(tdata1)) + ' min ' + str(np.min(tdata1)))
            # logger.debug('   max ' + str(np.max(tdata2)) + ' min ' + str(np.min(tdata2)))
            # logger.debug('gamma ' + str(gamma))

            # import sed3
            # ed = sed3.show_slices(tdata1)
            # ed = sed3.show_slices(tdata2)
            del tdata1u
            del tdata2u
            del a1
            del a2

        # if np.any(tdata1 < 0) or np.any(tdata2 <0):
        #     logger.error("Problem with tlinks. Likelihood is < 0")

        # if self.debug_images:
        #     self.__show_debug_tdata_images(tdata1, tdata2, suptitle="likelihood")
        return tdata1, tdata2
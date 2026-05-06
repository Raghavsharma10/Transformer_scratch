def spiceinit(self):
        """Perform either normal spiceinit or one for ringdata.

        Note how Python name-spacing can distinguish between the method
        and the function with the same name. `spiceinit` from the outer
        namespace is the one imported from pysis.
        """
        shape = "ringplane" if self.is_ring_data else None
        spiceinit(from_=self.pm.raw_cub, cksmithed="yes", spksmithed="yes", shape=shape)
        logger.info("spiceinit done.")
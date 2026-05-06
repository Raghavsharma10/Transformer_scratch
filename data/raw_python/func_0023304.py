def sky_fraction(self):
        """
        Sky fraction covered by the MOC
        """
        pix_id = self._best_res_pixels()
        nb_pix_filled = pix_id.size
        return nb_pix_filled / float(3 << (2*(self.max_order + 1)))
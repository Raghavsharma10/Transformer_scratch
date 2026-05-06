def sky2pix(self, skypos):
        """
        Get the pixel coordinates for a given sky position (degrees).

        Parameters
        ----------
        skypos : (float,float)
            ra,dec position in degrees.

        Returns
        -------
        x,y : float
            Pixel coordinates.

        """
        skybox = [skypos, skypos]
        pixbox = self.wcs.all_world2pix(skybox, 1)
        return [float(pixbox[0][0]), float(pixbox[0][1])]
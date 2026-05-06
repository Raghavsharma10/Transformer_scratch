def _calc_uca_section_proportion(self, data, dX, dY, direction, flats):
        """
        Given the direction, figure out which nodes the drainage will go
        toward, and what proportion of the drainage goes to which node
        """
        shp = np.array(data.shape) - 1

        facets = self.facets
        adjust = self.ang_adj[:, 1]

        d1, d2, theta = _get_d1_d2(dX, dY, 0, facets[0][1], facets[0][2], shp)
        if dX.size > 1:
            theta = np.row_stack((theta[0, :], theta, theta[-1, :]))
        # Which quadrant am I in?
        section = ((direction / np.pi * 2.0) // 1).astype('int8') # TODO DTYPE
        # Gets me in the quadrant
        quadrant = (direction - np.pi / 2.0 * section)
        proportion = np.full_like(quadrant, np.nan)
        # Now which section within the quadrant
        section = section * 2 \
            + (quadrant > theta.repeat(data.shape[1], 1)) * (section % 2 == 0) \
            + (quadrant > (np.pi/2 - theta.repeat(data.shape[1], 1))) \
            * (section % 2 == 1)  # greater than because of ties resolution b4

        # %%    Calculate proportion
        # As a side note, it's crazy to me how:
        #    _get_d1_d2 needs to use indices 0, 3, 4, 7,
        #    section uses even/odd (i.e. % 2)
        #    proportion uses indices (0, 1, 4, 5) {ALl of them different! ARG!}
        I1 = (section == 0) | (section == 1) | (section == 4) | (section == 5)
    #    I1 = section % 2 == 0
        I = I1 & (quadrant <= theta.repeat(data.shape[1], 1))
        proportion[I] = quadrant[I] / theta.repeat(data.shape[1], 1)[I]
        I = I1 & (quadrant > theta.repeat(data.shape[1], 1))
        proportion[I] = (quadrant[I] - theta.repeat(data.shape[1], 1)[I]) \
            / (np.pi / 2 - theta.repeat(data.shape[1], 1)[I])
        I = (~I1) & (quadrant <= (np.pi / 2 - theta.repeat(data.shape[1], 1)))
        proportion[I] = (quadrant[I]) \
            / (np.pi / 2 - theta.repeat(data.shape[1], 1)[I])
        I = (~I1) & (quadrant > (np.pi / 2 - theta.repeat(data.shape[1], 1)))
        proportion[I] = (quadrant[I] - (np.pi / 2 - theta.repeat(data.shape[1], 1)[I])) \
            / (theta.repeat(data.shape[1], 1)[I])
        # %%Finish Proportion Calculation
        section[flats] = FLAT_ID_INT
        proportion[flats] = FLAT_ID

        section[section == 8] = 0  # Fence-post error correction
        proportion = (1 + adjust[section]) / 2.0 - adjust[section] * proportion

        return section, proportion
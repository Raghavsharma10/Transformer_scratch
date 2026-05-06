def _mk_connectivity(self, section, i12, j1, j2):
        """
        Helper function for _mk_adjacency_matrix. Calculates the drainage
        neighbors and proportions based on the direction. This deals with
        non-flat regions in the image. In this case, each pixel can only
        drain to either 1 or two neighbors.
        """
        shp = np.array(section.shape) - 1

        facets = self.facets

        for ii, facet in enumerate(facets):
            e1 = facet[1]
            e2 = facet[2]
            I = section[1:-1, 1:-1] == ii
            j1[1:-1, 1:-1][I] = i12[1 + e1[0]:shp[0] + e1[0],
                                    1 + e1[1]:shp[1] + e1[1]][I]
            j2[1:-1, 1:-1][I] = i12[1 + e2[0]:shp[0] + e2[0],
                                    1 + e2[1]:shp[1] + e2[1]][I]

        # Now do the edges
        # left edge
        slc0 = [slice(1, -1), slice(0, 1)]
        for ind in [0, 1, 6, 7]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            I = section[slc0] == ind
            j1[slc0][I] = i12[1 + e1[0]:shp[0] + e1[0], e1[1]][I.ravel()]
            j2[slc0][I] = i12[1 + e2[0]:shp[0] + e2[0], e2[1]][I.ravel()]

        # right edge
        slc0 = [slice(1, -1), slice(-1, None)]
        for ind in [2, 3, 4, 5]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            I = section[slc0] == ind
            j1[slc0][I] = i12[1 + e1[0]:shp[0] + e1[0],
                              shp[1] + e1[1]][I.ravel()]
            j2[slc0][I] = i12[1 + e2[0]:shp[0] + e2[0],
                              shp[1] + e2[1]][I.ravel()]

        # top edge
        slc0 = [slice(0, 1), slice(1, -1)]
        for ind in [4, 5, 6, 7]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            I = section[slc0] == ind
            j1[slc0][I] = i12[e1[0], 1 + e1[1]:shp[1] + e1[1]][I.ravel()]
            j2[slc0][I] = i12[e2[0], 1 + e2[1]:shp[1] + e2[1]][I.ravel()]

        # bottom edge
        slc0 = [slice(-1, None), slice(1, -1)]
        for ind in [0, 1, 2, 3]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            I = section[slc0] == ind
            j1[slc0][I] = i12[shp[0] + e1[0],
                              1 + e1[1]:shp[1] + e1[1]][I.ravel()]
            j2[slc0][I] = i12[shp[0] + e2[0],
                              1 + e2[1]:shp[1] + e2[1]][I.ravel()]

        # top-left corner
        slc0 = [slice(0, 1), slice(0, 1)]
        for ind in [6, 7]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            if section[slc0] == ind:
                j1[slc0] = i12[e1[0], e1[1]]
                j2[slc0] = i12[e2[0], e2[1]]

        # top-right corner
        slc0 = [slice(0, 1), slice(-1, None)]
        for ind in [4, 5]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            if section[slc0] == ind:
                j1[slc0] = i12[e1[0], shp[1] + e1[1]]
                j2[slc0] = i12[e2[0], shp[1] + e2[1]]

        # bottom-left corner
        slc0 = [slice(-1, None), slice(0, 1)]
        for ind in [0, 1]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            if section[slc0] == ind:
                j1[slc0] = i12[shp[0] + e1[0], e1[1]]
                j2[slc0] = i12[shp[0] + e2[0], e2[1]]

        # bottom-right corner
        slc0 = [slice(-1, None), slice(-1, None)]
        for ind in [2, 3]:
            e1 = facets[ind][1]
            e2 = facets[ind][2]
            if section[slc0] == ind:
                j1[slc0] = i12[e1[0] + shp[0], shp[1] + e1[1]]
                j2[slc0] = i12[e2[0] + shp[0], shp[1] + e2[1]]

        return j1, j2
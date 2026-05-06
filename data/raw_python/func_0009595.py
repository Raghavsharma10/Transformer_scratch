def correctGrid(self, img, grid):
        '''
        grid -> array of polylines=((p0x,p0y),(p1x,p1y),,,)
        '''

        self.img = imread(img)
        h = self.homography  # TODO: cleanup only needed to get newBorder attr.

        if self.opts['do_correctIntensity']:
            self.img = self.img / self._getTiltFactor(self.img.shape)

        s0, s1 = grid.shape[:2]
        n0, n1 = s0 - 1, s1 - 1

        snew = self._newBorders
        b = self.opts['border']

        sx, sy = (snew[0] - 2 * b) // n0, (snew[1] - 2 * b) // n1

        out = np.empty(snew[::-1], dtype=self.img.dtype)

        def warp(ix, iy, objP, outcut):
            shape = outcut.shape[::-1]
            quad = grid[ix:ix + 2,
                        iy:iy + 2].reshape(4, 2)[np.array([0, 2, 3, 1])]
            hcell = cv2.getPerspectiveTransform(
                quad.astype(np.float32), objP)
            cv2.warpPerspective(self.img, hcell, shape, outcut,
                                flags=cv2.INTER_LANCZOS4,
                                **self.opts['cv2_opts'])
            return quad

        objP = np.array([[0, 0],
                         [sx, 0],
                         [sx, sy],
                         [0, sy]], dtype=np.float32)
        # INNER CELLS
        for ix in range(1, n0 - 1):
            for iy in range(1, n1 - 1):
                sub = out[iy * sy + b: (iy + 1) * sy + b,
                          ix * sx + b: (ix + 1) * sx + b]
#                 warp(ix, iy, objP, sub)

                shape = sub.shape[::-1]
                quad = grid[ix:ix + 2,
                            iy:iy + 2].reshape(4, 2)[np.array([0, 2, 3, 1])]
#                 print(quad, objP)

                hcell = cv2.getPerspectiveTransform(
                    quad.astype(np.float32), objP)
                cv2.warpPerspective(self.img, hcell, shape, sub,
                                    flags=cv2.INTER_LANCZOS4,
                                    **self.opts['cv2_opts'])

#         return out
        # TOP CELLS
        objP[:, 1] += b
        for ix in range(1, n0 - 1):
            warp(ix, 0, objP, out[: sy + b,
                                  ix * sx + b: (ix + 1) * sx + b])
        # BOTTOM CELLS
        objP[:, 1] -= b
        for ix in range(1, n0 - 1):
            iy = (n1 - 1)
            y = iy * sy + b
            x = ix * sx + b
            warp(ix, iy, objP, out[y: y + sy + b, x: x + sx])
        # LEFT CELLS
        objP[:, 0] += b
        for iy in range(1, n1 - 1):
            y = iy * sy + b
            warp(0, iy, objP, out[y: y + sy, : sx + b])
        # RIGHT CELLS
        objP[:, 0] -= b
        ix = (n0 - 1)
        x = ix * sx + b
        for iy in range(1, n1 - 1):
            y = iy * sy + b
            warp(ix, iy, objP, out[y: y + sy, x: x + sx + b])
        # BOTTOM RIGHT CORNER
        warp(n0 - 1, n1 - 1, objP, out[-sy - b - 1:, x: x + sx + b])
#         #TOP LEFT CORNER
        objP += (b, b)
        warp(0, 0, objP, out[0: sy + b, 0: sx + b])
        # TOP RIGHT CORNER
        objP[:, 0] -= b
#         x = (n0-1)*sx+b
        warp(n0 - 1, 0, objP, out[: sy + b, x: x + sx + b])
#         #BOTTOM LEFT CORNER
        objP += (b, -b)
        warp(0, n1 - 1, objP, out[-sy - b - 1:, : sx + b])
        return out
def distort(self, img, rotX=0, rotY=0, quad=None):
        '''
        Apply perspective distortion ion self.img
        angles are in DEG and need to be positive to fit into image

        '''
        self.img = imread(img)
        # fit old image to self.quad:
        corr = self.correct(self.img)

        s = self.img.shape
        if quad is None:
            wquad = (self.quad - self.quad.mean(axis=0)).astype(float)

            win_width = s[1]
            win_height = s[0]
            # project quad:
            for n, q in enumerate(wquad):
                p = Point3D(q[0], q[1], 0).rotateX(-rotX).rotateY(-rotY)
                p = p.project(win_width, win_height, s[1], s[1])
                wquad[n] = (p.x, p.y)
            wquad = sortCorners(wquad)
            # scale result so that longest side of quad and wquad are equal
            w = wquad[:, 0].max() - wquad[:, 0].min()
            h = wquad[:, 1].max() - wquad[:, 1].min()
            scale = min(s[1] / w, s[0] / h)
            # scale:
            wquad = (wquad * scale).astype(int)
        else:
            wquad = sortCorners(quad)
        wquad -= wquad.min(axis=0)

        lx = corr.shape[1]
        ly = corr.shape[0]

        objP = np.array([
            [0, 0],
            [lx, 0],
            [lx, ly],
            [0, ly],
        ], dtype=np.float32)

        homography = cv2.getPerspectiveTransform(
            wquad.astype(np.float32), objP)
        # distort corr:
        w = wquad[:, 0].max() - wquad[:, 0].min()
        h = wquad[:, 1].max() - wquad[:, 1].min()
        #(int(w),int(h))
        dist = cv2.warpPerspective(corr, homography, (int(w), int(h)),
                                   flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP)

        # move middle of dist to middle of the old quad:
        bg = np.zeros(shape=s)
        rmn = (bg.shape[0] / 2, bg.shape[1] / 2)

        ss = dist.shape
        mn = (ss[0] / 2, ss[1] / 2)  # wquad.mean(axis=0)
        ref = (int(rmn[0] - mn[0]), int(rmn[1] - mn[1]))

        bg[ref[0]:ss[0] + ref[0], ref[1]:ss[1] + ref[1]] = dist

        # finally move quad into right position:
        self.quad = wquad
        self.quad += (ref[1], ref[0])
        self.img = bg
        self._homography = None
        self._poseFromQuad()

        if self.opts['do_correctIntensity']:
            tf = self.tiltFactor()
            if self.img.ndim == 3:
                for col in range(self.img.shape[2]):
                    self.img[..., col] *= tf
            else:
                #                 tf = np.tile(tf, (1,1,self.img.shape[2]))
                self.img = self.img * tf

        return self.img
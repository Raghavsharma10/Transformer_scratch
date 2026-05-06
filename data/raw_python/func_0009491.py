def addImg(self, img, roi=None):
        '''
        img - background, flat field, ste corrected image
        roi - [(x1,y1),...,(x4,y4)] -  boundaries where points are
        '''
        self.img = imread(img, 'gray')
        s0, s1 = self.img.shape

        if roi is None:
            roi = ((0, 0), (s0, 0), (s0, s1), (0, s1))

        k = self.kernel_size
        hk = k // 2

        # mask image
        img2 = self.img.copy()  # .astype(int)

        mask = np.zeros(self.img.shape)
        cv2.fillConvexPoly(mask, np.asarray(roi, dtype=np.int32), color=1)
        mask = mask.astype(bool)
        im = img2[mask]

        bg = im.mean()  # assume image average with in roi == background
        mask = ~mask
        img2[mask] = -1

        # find points from local maxima:
        self.points = np.zeros(shape=(self.max_points, 2), dtype=int)
        thresh = 0.8 * bg + 0.2 * im.max()

        _findPoints(img2, thresh, self.min_dist, self.points)
        self.points = self.points[:np.argmin(self.points, axis=0)[0]]

        # correct point position, to that every point is over max value:
        for n, p in enumerate(self.points):
            sub = self.img[p[1] - hk:p[1] + hk + 1, p[0] - hk:p[0] + hk + 1]
            i, j = np.unravel_index(np.nanargmax(sub), sub.shape)
            self.points[n] += [j - hk, i - hk]

        # remove points that are too close to their neighbour or the border
        mask = maximum_filter(mask, hk)
        i = np.ones(self.points.shape[0], dtype=bool)
        for n, p in enumerate(self.points):
            if mask[p[1], p[0]]:  # too close to border
                i[n] = False
            else:
                # too close to other points
                for pp in self.points[n + 1:]:
                    if norm(p - pp) < hk + 1:
                        i[n] = False
        isum = i.sum()
        ll = len(i) - isum
        print('found %s points' % isum)
        if ll:
            print(
                'removed %s points (too close to border or other points)' %
                ll)
            self.points = self.points[i]

#         self.n_points += len(self.points)

        # for finding best peak position:
#         def fn(xy,cx,cy):#par
#             (x,y) = xy
#             return 1-(((x-cx)**2 + (y-cy)**2)*(1/8)).flatten()

#         x,y = np.mgrid[-2:3,-2:3]
#         x = x.flatten()
#         y = y.flatten()
        # for shifting peak:
        xx, yy = np.mgrid[0:k, 0:k]
        xx = xx.astype(float)
        yy = yy.astype(float)

        self.subs = []


#         import pylab as plt
#         plt.figure(20)
#         img = self.drawPoints()
#         plt.imshow(img, interpolation='none')
# #                 plt.figure(21)
# #                 plt.imshow(sub2, interpolation='none')
#         plt.show()

        #thresh = 0.8*bg + 0.1*im.max()
        for i, p in enumerate(self.points):
            sub = self.img[p[1] - hk:p[1] + hk + 1,
                           p[0] - hk:p[0] + hk + 1].astype(float)
            sub2 = sub.copy()

            mean = sub2.mean()
            mx = sub2.max()
            sub2[sub2 < 0.5 * (mean + mx)] = 0  # only select peak
            try:
                # SHIFT SUB ARRAY to align peak maximum exactly in middle:
                    # only eval a 5x5 array in middle of sub:
                # peak = sub[hk-3:hk+4,hk-3:hk+4]#.copy()

                #                 peak -= peak.min()
                #                 peak/=peak.max()
                #                 peak = peak.flatten()
                    # fit paraboloid to get shift in x,y:
                #                 p, _ = curve_fit(fn, (x,y), peak, (0,0))
                c0, c1 = center_of_mass(sub2)

#                 print (p,c0,c1,hk)

                #coords = np.array([xx+p[0],yy+p[1]])
                coords = np.array([xx + (c0 - hk), yy + (c1 - hk)])

                #print (c0,c1)

                #import pylab as plt
                #plt.imshow(sub2, interpolation='none')

                # shift array:
                sub = map_coordinates(sub, coords,
                                      mode='nearest').reshape(k, k)
                # plt.figure(2)
                #plt.imshow(sub, interpolation='none')
                # plt.show()

                #normalize:
                bg = 0.25* (  sub[0].mean()   + sub[-1].mean() 
                            + sub[:,0].mean() + sub[:,-1].mean())
                 
                sub-=bg
                sub /= sub.max()

#                 import pylab as plt
#                 plt.figure(20)
#                 plt.imshow(sub, interpolation='none')
# #                 plt.figure(21)
# #                 plt.imshow(sub2, interpolation='none')
#                 plt.show()

                self._psf += sub

                if self.calc_std:
                    self.subs.append(sub)
            except ValueError:
                pass
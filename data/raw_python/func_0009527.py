def findHomography(self, img, drawMatches=False):
        '''
        Find homography of the image through pattern
        comparison with the base image
        '''
        print("\t Finding points...")
        # Find points in the next frame
        img = self._prepareImage(img)
        features, descs = self.detector.detectAndCompute(img, None)

        ######################
        # TODO: CURRENTLY BROKEN IN OPENCV3.1 - WAITNG FOR NEW RELEASE 3.2
#         matches = self.matcher.knnMatch(descs,#.astype(np.float32),
#                                         self.base_descs,
#                                         k=3)
#         print("\t Match Count: ", len(matches))
#         matches_subset = self._filterMatches(matches)

        # its working alternative (for now):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches_subset = bf.match(descs, self.base_descs)

        ######################
#         matches = bf.knnMatch(descs,self.base_descs, k=2)
#         # Apply ratio test
#         matches_subset = []
#         medDist = np.median([m.distance for m in matches])
#         matches_subset = [m for m in matches if m.distance < medDist]
#         for m in matches:
#             print(m.distance)
#         for m,n in matches:
#             if m.distance < 0.75*n.distance:
#                 matches_subset.append([m])

        if not len(matches_subset):
            raise Exception('no matches found')
        print("\t Filtered Match Count: ", len(matches_subset))

        distance = sum([m.distance for m in matches_subset])
        print("\t Distance from Key Image: ", distance)

        averagePointDistance = distance / (len(matches_subset))
        print("\t Average Distance: ", averagePointDistance)

        kp1 = []
        kp2 = []

        for match in matches_subset:
            kp1.append(self.base_features[match.trainIdx])
            kp2.append(features[match.queryIdx])

        # /self._fH #scale with _fH, if image was resized

        p1 = np.array([k.pt for k in kp1])
        p2 = np.array([k.pt for k in kp2])  # /self._fH

        H, status = cv2.findHomography(p1, p2,
                                       cv2.RANSAC,  # method
                                       5.0  # max reprojection error (1...10)
                                       )
        if status is None:
            raise Exception('no homography found')
        else:
            inliers = np.sum(status)
            print('%d / %d  inliers/matched' % (inliers, len(status)))
            inlierRatio = inliers / len(status)
            if self.minInlierRatio > inlierRatio or inliers < self.minInliers:
                raise Exception('bad fit!')

        # scale with _fH, if image was resized
        # see
        # http://answers.opencv.org/question/26173/the-relationship-between-homography-matrix-and-scaling-images/
        s = np.eye(3, 3)
        s[0, 0] = 1 / self._fH
        s[1, 1] = 1 / self._fH
        H = s.dot(H).dot(np.linalg.inv(s))

        if drawMatches:
            #             s0,s1 = img.shape
            #             out = np.empty(shape=(s0,s1,3), dtype=np.uint8)
            img = draw_matches(self.base8bit, self.base_features, img, features,
                               matches_subset[:20],  # None,#out,
                               # flags=2
                               thickness=5
                               )

        return (H, inliers, inlierRatio, averagePointDistance,
                img, features,
                descs, len(matches_subset))
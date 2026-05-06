def setReference(self, ref):
        '''
        ref  ... either quad, grid, homography or reference image

        quad --> list of four image points(x,y) marking the edges of the quad
               to correct
        homography --> h. matrix to correct perspective distortion
        referenceImage --> image of same object without perspective distortion
        '''
#         self.maps = {}
        self.quad = None
#         self.refQuad = None
        self._camera_position = None
        self._homography = None
        self._homography_is_fixed = True
#         self.tvec, self.rvec = None, None
        self._pose = None

        # evaluate input:
        if isinstance(ref, np.ndarray) and ref.shape == (3, 3):
            # REF IS HOMOGRAPHY
            self._homography = ref
            # REF IS QUAD
        elif len(ref) == 4:
            self.quad = sortCorners(ref)

            # TODO: cleanup # only need to call once - here
            o = self.obj_points  # no property any more

            # REF IS IMAGE
        else:
            self.ref = imread(ref)
#             self._refshape = ref.shape[:2]
            self.pattern = PatternRecognition(self.ref)
            self._homography_is_fixed = False
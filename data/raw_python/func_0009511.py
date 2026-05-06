def addPoints(self, points, board_size=None):
        '''
        add corner points directly instead of extracting them from
        image
        points = ( (0,1), (...),... ) [x,y]
        '''
        self.opts['foundPattern'].append(True)
        self.findCount += 1
        if board_size is not None:
            self.objpoints.append(self._mkObjPoints(board_size))
        else:
            self.objpoints.append(self.objp)
        s0 = points.shape[0]

        self.opts['imgPoints'].append(np.asarray(points).reshape(
            s0, 1, 2).astype(np.float32))
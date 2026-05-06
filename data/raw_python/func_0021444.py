def euler_angles_and_eigenframes(self):
        '''Calculate the Euler angles only if the rotation matrix
           (eigenframe) has positive determinant.'''
        signs = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1],
                          [1, 1, -1], [-1, -1, 1], [-1, 1, -1],
                          [1, -1, -1], [-1, -1, -1]])
        eulangs = []
        eigframes = []
        for i, sign in enumerate(signs):
            eigframe = np.dot(self.eigvecs, np.diag(sign))
            if np.linalg.det(eigframe) > 1e-4:
                eigframes.append(np.array(eigframe))
                eulangs.append(np.array(
                    transformations.euler_from_matrix(eigframe, axes='szyz')))

        self.eigframes = np.array(eigframes)
        # The sign has to be inverted to be consistent with ORCA and EasySpin.
        self.eulangs = -np.array(eulangs)
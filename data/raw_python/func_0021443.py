def diagonalize(self):
        '''Diagonalize the tensor.'''
        self.eigvals, self.eigvecs = np.linalg.eig(
            (self.tensor.transpose() + self.tensor) / 2.0)
        self.eigvals = np.diag(np.dot(
            np.dot(self.eigvecs.transpose(), self.tensor), self.eigvecs))
def random_draw(self, size=None):
        """Draw random samples of the hyperparameters.
        
        The outputs of the two priors are stacked vertically.
        
        Parameters
        ----------
        size : None, int or array-like, optional
            The number/shape of samples to draw. If None, only one sample is
            returned. Default is None.
        """
        draw_1 = self.p1.random_draw(size=size)
        draw_2 = self.p2.random_draw(size=size)
        
        if draw_1.ndim == 1:
            return scipy.hstack((draw_1, draw_2))
        else:
            return scipy.vstack((draw_1, draw_2))
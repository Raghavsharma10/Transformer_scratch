def gradient(self, y_true, y_pred):
        """Returns the gradient of the L1 loss with respect to each prediction.

        Example:
            >>> import starboost as sb
            >>> y_true = [0, 0, 1]
            >>> y_pred = [0.3, 0, 0.8]
            >>> sb.losses.L1Loss().gradient(y_true, y_pred)
            array([ 1.,  0., -1.])
        """
        return np.sign(np.subtract(y_pred, y_true))
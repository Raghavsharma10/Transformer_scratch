def out_of_bag_mae(self):
        """
        Returns the mean absolute error for predictions on the out-of-bag
        samples.
        """
        if not self._out_of_bag_mae_clean:
            try:
                self._out_of_bag_mae = self.test(self.out_of_bag_samples)
                self._out_of_bag_mae_clean = True
            except NodeNotReadyToPredict:
                return
        return self._out_of_bag_mae.copy()
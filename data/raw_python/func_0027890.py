def wiggle(self,noiseLevel=.1):
        """Slightly changes value of every cell in the worksheet. Used for testing."""
        noise=(np.random.rand(*self.data.shape))-.5
        self.data=self.data+noise*noiseLevel
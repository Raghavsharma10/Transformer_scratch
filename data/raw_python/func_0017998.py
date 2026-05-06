def parameters(self, value):
        """Update the parameters. ``value`` must have the shape
        ``(weights, biases)``"""
        self.W = value[0] if isinstance(value[0], GPUArray) else \
          gpuarray.to_gpu(value[0])
        self.b = value[1] if isinstance(value[0], GPUArray) else \
          gpuarray.to_gpu(value[1])
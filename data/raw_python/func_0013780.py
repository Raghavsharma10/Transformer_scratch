def run(self, inputs, **kwargs):
        """Run model inference and return the result

        Parameters
        ----------
        inputs : numpy array
            input to run a layer on

        Returns
        -------
        params : numpy array
            result obtained after running the inference on mxnet
        """
        input_data = np.asarray(inputs[0], dtype='f')

        # create module, passing cpu context
        if self.device == 'CPU':
            ctx = mx.cpu()
        else:
            raise NotImplementedError("Only CPU context is supported for now")

        mod = mx.mod.Module(symbol=self.symbol, data_names=['input_0'], context=ctx,
                            label_names=None)
        mod.bind(for_training=False, data_shapes=[('input_0', input_data.shape)],
                 label_shapes=None)
        mod.set_params(arg_params=self.params, aux_params=None)

        # run inference
        batch = namedtuple('Batch', ['data'])

        mod.forward(batch([mx.nd.array(input_data)]))
        result = mod.get_outputs()[0].asnumpy()
        return [result]
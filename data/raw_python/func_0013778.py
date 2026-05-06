def _fix_bias_shape(self, op_name, inputs, attrs):
        """A workaround to reshape bias term to (1, num_channel)."""
        if (op_name == 'Add' or op_name == 'Mul') and (int(len(self._params)) > 0) and \
                ('broadcast' in attrs and attrs['broadcast'] == 1):
            assert len(list(inputs)) == 2
            bias_name = self._renames.get(inputs[1], inputs[1])
            bias = self._params[bias_name]
            assert len(bias.shape) == 1
            # reshape to (1, n)
            bias = mx.nd.array(bias.asnumpy().reshape((1, -1, 1, 1)))
            # broadcast_add expects shape with sym.variable
            self._nodes[bias_name] = mx.sym.Variable(name=bias_name, shape=bias.shape)
            self._params[bias_name] = bias
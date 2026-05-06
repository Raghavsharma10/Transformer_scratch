def _fix_bias(self, op, attrs, num_inputs):
        """A workaround for 'use_bias' attribute since onnx don't provide this attribute,
        we have to check the number of inputs to decide it."""
        if op not in [mx.sym.Convolution, mx.sym.Deconvolution, mx.sym.FullyConnected]:
            return attrs
        if num_inputs == 3:
            attrs['no_bias'] = False
        elif num_inputs == 2:
            attrs['no_bias'] = True
        else:
            raise ValueError("Unexpected number of inputs for: {}".format(op))
        return attrs
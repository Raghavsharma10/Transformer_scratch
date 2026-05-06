def _fix_channels(self, op, attrs, inputs):
        """A workaround for getting 'channels' or 'units' since onnx don't provide
        these attributes. We check the shape of weights provided to get the number.
        """
        if op not in [mx.sym.Convolution, mx.sym.Deconvolution, mx.sym.FullyConnected]:
            return attrs
        weight_name = self._renames[inputs[1]]
        if not weight_name in self._params:
            raise ValueError("Unable to get channels/units attr from onnx graph.")
        else:
            wshape = self._params[weight_name].shape
            assert len(wshape) >= 2, "Weights shape is invalid: {}".format(wshape)

            if op in [mx.sym.FullyConnected]:
                attrs['num_hidden'] = wshape[0]
            else:
                if op == mx.sym.Convolution:
                    # Weight shape for Conv and FC: (M x C x kH x kW) : M is number of
                    # feature maps/hidden  and C is number of channels
                    attrs['num_filter'] = wshape[0]
                elif op == mx.sym.Deconvolution:
                    # Weight shape for DeConv : (C x M x kH x kW) : M is number of
                    # feature maps/filters and C is number of channels
                    attrs['num_filter'] = wshape[1]
        return attrs
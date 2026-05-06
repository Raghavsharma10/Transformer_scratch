def _fix_pooling(self, op_name, inputs, new_attr):
        """onnx pooling operator supports asymmetrical padding
        Adding pad operator before pooling in mxnet to work with onnx"""
        pool_type = 'avg' if op_name == 'AveragePool' else 'max'
        stride = new_attr.get('strides')
        kernel = new_attr.get('kernel_shape')
        padding = new_attr.get('pads')
        pad_width = (0, 0, 0, 0) + _pad_sequence_fix(padding, len(kernel))
        new_pad_op = mx.sym.pad(inputs[0], mode='constant', pad_width=pad_width)
        new_pooling_op = mx.sym.Pooling(new_pad_op, pool_type=pool_type,
                                        stride=stride, kernel=kernel)
        return new_pooling_op
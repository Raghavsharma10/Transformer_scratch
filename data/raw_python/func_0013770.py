def _fix_slice(self, inputs, new_attr):
        """onnx slice provides slicing on multiple axis. Adding multiple slice_axis operator
        for multiple axes from mxnet"""
        begin = new_attr.get('begin')
        end = new_attr.get('end')
        axes = new_attr.get('axis', tuple(range(len(begin))))
        slice_op = mx.sym.slice_axis(inputs[0], axis=axes[0], begin=begin[0], end=end[0])
        if len(axes) > 1:
            for i, axis in enumerate(axes):
                slice_op = mx.sym.slice_axis(slice_op, axis=axis, begin=begin[i], end=end[i])
        return slice_op
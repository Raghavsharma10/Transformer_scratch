def _fix_squeeze(self, inputs, new_attr):
        """
        MXNet doesnt have a squeeze operator.
        Using "split" to perform similar operation.
        "split" can be slower compared to "reshape".
         This can have performance impact.
         TODO: Remove this implementation once mxnet adds the support.
        """
        axes = new_attr.get('axis')
        op = mx.sym.split(inputs[0], axis=axes[0], num_outputs=1, squeeze_axis=1)
        for i in axes[1:]:
            op = mx.sym.split(op, axis=i-1, num_outputs=1, squeeze_axis=1)
        return op
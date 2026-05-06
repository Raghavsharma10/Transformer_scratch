def _fix_max_min(self, op_name, inputs):
        """ MXNet maximum/minimum compares only two symbols at a time.
            ONNX can send more than two to compare.
            Breaking into multiple mxnet ops to compare two symbols at a time"""
        if len(inputs) > 1:
            if op_name == 'Max':
                op = mx.sym.maximum(inputs[0], inputs[1])
                for ip in inputs[2:]:
                    op = mx.sym.maximum(op, ip)
            elif op_name == 'Min':
                op = mx.sym.minimum(inputs[0], inputs[1])
                for ip in inputs[2:]:
                    op = mx.sym.minimum(op, ip)
        else:
            op = inputs[0]
        return op
def _fix_gemm(self, op_name, inputs, old_attr):
        """Using FullyConnected operator in place of linalg_gemm to perform same operation"""
        op = getattr(mx.sym, op_name, None)
        alpha = float(old_attr.get('alpha', 1.0))
        beta = float(old_attr.get('beta', 1.0))
        transA = int(old_attr.get('transA', 0))
        transB = int(old_attr.get('transB', 0))
        if transA:
            inputs[0] = mx.sym.transpose(inputs[0], axes=(1, 0))
        if not transB:
            inputs[1] = mx.sym.transpose(inputs[1], axes=(1, 0))
        new_inputs = [alpha*inputs[0], inputs[1], beta*inputs[2]]
        new_attr = {'num_hidden' : self._params[inputs[2].name].shape[0]}
        return op, new_inputs, new_attr
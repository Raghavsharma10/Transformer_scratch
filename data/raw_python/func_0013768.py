def from_onnx(self, graph):
        """Construct symbol from onnx graph.
        The inputs from onnx graph is vague, only providing "1", "2"...
        For convenience, we rename the `real` input names to "input_0",
        "input_1"... And renaming parameters to "param_0", "param_1"...

        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph

        Returns
        -------
        sym :mx.sym
            The returned mxnet symbol
        params : dict
            A dict of name: mx.nd.array pairs, used as pretrained weights
        """
        # parse network inputs, aka parameters
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            self._params[init_tensor.name] = self._parse_array(init_tensor)

        # converting GraphProto message
        for i in graph.input:
            if i.name in self._params:
                # i is a param instead of input
                name_param = 'param_{}'.format(self._num_param)
                self._num_param += 1
                self._params[name_param] = self._params.pop(i.name)
                self._nodes[name_param] = mx.sym.Variable(name=name_param,
                                                          shape=self._params[name_param].shape)
                self._renames[i.name] = name_param
            else:
                name_input = 'input_{}'.format(self._num_input)
                self._num_input += 1
                self._nodes[name_input] = mx.sym.Variable(name=name_input)
                self._renames[i.name] = name_input

        # constructing nodes, nodes are stored as directed acyclic graph
        # converting NodeProto message
        for node in graph.node:
            op_name = node.op_type
            node_name = node.name.strip()
            node_name = node_name if node_name else None
            onnx_attr = self._parse_attr(node.attribute)
            new_op, mx_attr = _convert_operator(op_name, onnx_attr)
            inputs = [self._nodes[self._renames.get(i, i)] for i in node.input]

            # some workarounds for inconsistencies in onnx and mxnet conventions.
            mx_attr = self._fix_bias(new_op, mx_attr, len(inputs))
            mx_attr = self._fix_channels(new_op, mx_attr, list(node.input))
            self._fix_bias_shape(node.op_type, node.input, onnx_attr)

            # calling again to get new symbols after some workarounds
            inputs = [self._nodes[self._renames.get(i, i)] for i in node.input]

            # onnx's Gemm operator also supports broadcasting C input which
            # mxnet's equivalent linalg_gemm doesn't. So using combination of
            # transpose and FullyConnected operators.
            if op_name == 'Gemm':
                new_op, inputs, mx_attr = self._fix_gemm('FullyConnected', inputs, onnx_attr)

            # onnx slice works on multiple axes whereas mxnet's slice_axis is for single axis
            if op_name == 'Slice':
                op = self._fix_slice(inputs, mx_attr)
            elif op_name == 'AveragePool' and onnx_attr.get('pads') is not None or \
                                    op_name == 'MaxPool' and onnx_attr.get('pads') is not None:
                op = self._fix_pooling(op_name, inputs, onnx_attr)
            elif op_name == 'Squeeze':
                op = self._fix_squeeze(inputs, mx_attr)
            elif op_name == 'Max' or op_name == 'Min':
                op = self._fix_max_min(op_name, inputs)
            elif node_name is None:
                op = new_op(*inputs, **mx_attr)
            else:
                op = new_op(name=node_name, *inputs, **mx_attr)

            node_output = self._fix_outputs(op_name, node.output)

            assert len(node_output) == len(op.list_outputs()), (
                "Number of output mismatch {} vs {} in {}.".format(
                    len(node_output), len(op.list_outputs()), op_name))
            for k, i in zip(list(node_output), range(len(node_output))):
                self._nodes[k] = op[i]
        # now return the outputs
        out = [self._nodes[i.name] for i in graph.output]
        if len(out) > 1:
            out = mx.sym.Group(out)
        else:
            out = out[0]
        return out, self._params
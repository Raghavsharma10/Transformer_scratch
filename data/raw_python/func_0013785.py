def run_node(cls, node, inputs, device='CPU'):  # pylint: disable=arguments-differ
        """Running individual node inference on mxnet engine and
        return the result to onnx test infrastructure.

        Parameters
        ----------
        node   : onnx node object
            loaded onnx node (individual layer)
        inputs : numpy array
            input to run a node on
        device : 'CPU'
            device to run a node on

        Returns
        -------
        params : numpy array
            result obtained after running the operator
        """
        graph = GraphProto()
        sym, _ = graph.from_onnx(MXNetBackend.make_graph(node, inputs))
        data_names = [i for i in sym.get_internals().list_inputs()]
        data_shapes = []
        reduce_op_types = set(['ReduceMin', 'ReduceMax', 'ReduceMean',
                               'ReduceProd', 'ReduceSum', 'Slice', 'Pad',
                               'Squeeze', 'Upsample', 'Reshape', 'Conv', 'ConvTranspose'])

        # Adding extra dimension of batch_size 1 if the batch_size is different for multiple inputs.
        for idx, input_name in enumerate(data_names):
            batch_size = 1
            if len(inputs[idx].shape) < 4 and len(inputs) > 1 and \
                            len(set(x.shape[0] for x in inputs)) != 1:
                tuples = ((batch_size,), inputs[idx].shape)
                new_shape = sum(tuples, ())
                data_shapes.append((input_name, new_shape))
            else:
                data_shapes.append((input_name, inputs[idx].shape))

        # create module, passing cpu context
        if device == 'CPU':
            ctx = mx.cpu()
        else:
            raise NotImplementedError("Only CPU context is supported for now")

        # create a module
        mod = mx.mod.Module(symbol=sym, data_names=data_names, context=ctx, label_names=None)
        mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)

        # initializing parameters for calculating result of each individual node
        mod.init_params()

        data_forward = []
        for idx, input_name in enumerate(data_names):
            # slice and pad operator tests needs 1 less dimension in forward pass
            # otherwise it will throw an error.
            # for squeeze operator, need to retain shape of input as provided
            val = inputs[idx]
            if node.op_type in reduce_op_types:
                data_forward.append(mx.nd.array(val))
            else:
                data_forward.append(mx.nd.array([val]))

        mod.forward(mx.io.DataBatch(data_forward))
        result = mod.get_outputs()[0].asnumpy()
        if node.op_type in reduce_op_types:
            return [result]
        return result
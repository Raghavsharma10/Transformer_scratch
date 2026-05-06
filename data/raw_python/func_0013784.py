def make_graph(node, inputs):
        """ Created ONNX GraphProto from node"""
        initializer = []
        tensor_input_info = []
        tensor_output_info = []

        # Adding input tensor info.
        for index in range(len(node.input)):
            tensor_input_info.append(
                helper.make_tensor_value_info(str(node.input[index]), TensorProto.FLOAT, [1]))

            # Creating an initializer for Weight params.
            # Assumes that weight params is named as 'W'.
            # TODO: Handle multiple weight params.
            # TODO: Add for "bias" if needed
            if node.input[index] == 'W':
                dim = inputs[index].shape
                param_tensor = helper.make_tensor(
                    name=node.input[index],
                    data_type=TensorProto.FLOAT,
                    dims=dim,
                    vals=inputs[index].flatten())

                initializer.append(param_tensor)

        # Adding output tensor info.
        for index in range(len(node.output)):
            tensor_output_info.append(
                helper.make_tensor_value_info(str(node.output[index]), TensorProto.FLOAT, [1]))

        # creating graph proto object.
        graph_proto = helper.make_graph(
            [node],
            "test",
            tensor_input_info,
            tensor_output_info,
            initializer=initializer)

        return graph_proto
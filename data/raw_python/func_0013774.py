def _parse_array(self, tensor_proto):
        """Grab data in TensorProto and convert to numpy array."""
        try:
            from onnx.numpy_helper import to_array
        except ImportError as e:
            raise ImportError("Unable to import onnx which is required {}".format(e))
        np_array = to_array(tensor_proto).reshape(tuple(tensor_proto.dims))
        return mx.nd.array(np_array)
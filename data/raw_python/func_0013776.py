def _fix_outputs(self, op, outputs):
        """A workaround to handle dropout or similar operator that have more than one out
        in ONNX.
        """
        if op == 'Dropout':
            assert len(outputs) == 2, "ONNX have two outputs for dropout layer."
            outputs = outputs[:-1]
        return outputs
def assignrepr(self, prefix) -> str:
        """Return a string representation of the actual |anntools.ANN| object
        that is prefixed with the given string."""
        prefix = '%s%s(' % (prefix, self.name)
        blanks = len(prefix)*' '
        lines = [
            objecttools.assignrepr_value(
                self.nmb_inputs, '%snmb_inputs=' % prefix)+',',
            objecttools.assignrepr_tuple(
                self.nmb_neurons, '%snmb_neurons=' % blanks)+',',
            objecttools.assignrepr_value(
                self.nmb_outputs, '%snmb_outputs=' % blanks)+',',
            objecttools.assignrepr_list2(
                self.weights_input, '%sweights_input=' % blanks)+',']
        if self.nmb_layers > 1:
            lines.append(objecttools.assignrepr_list3(
                self.weights_hidden, '%sweights_hidden=' % blanks)+',')
        lines.append(objecttools.assignrepr_list2(
            self.weights_output, '%sweights_output=' % blanks)+',')
        lines.append(objecttools.assignrepr_list2(
            self.intercepts_hidden, '%sintercepts_hidden=' % blanks)+',')
        lines.append(objecttools.assignrepr_list(
            self.intercepts_output, '%sintercepts_output=' % blanks)+')')
        return '\n'.join(lines)
def remove_leading_spaces(self, input_model):
        """
        This function is a part of the model  post-processing pipeline. It
        removes leading spaces from an extracted module; depending on the
        formatting of the draft/rfc text, may have multiple spaces prepended
        to each line. The function also determines the length of the longest
        line in the module - this value can be used by later stages of the
        model post-processing pipeline.
        :param input_model: The YANG model to be processed
        :return: YANG model lines with leading spaces removed
        """
        leading_spaces = 1024
        output_model = []
        for mline in input_model:
            line = mline[0]
            if line.rstrip(' \r\n') != '':
                leading_spaces = min(leading_spaces, len(line) - len(line.lstrip(' ')))
                output_model.append([line[leading_spaces:], mline[1]])

                line_len = len(line[leading_spaces:])
                if line_len > self.max_line_len:
                    self.max_line_len = line_len
            else:
                output_model.append(['\n', mline[1]])
        return output_model
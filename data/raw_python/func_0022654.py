def remove_extra_empty_lines(self, input_model):
        """
        Removes superfluous newlines from a YANG model that was extracted
        from a draft or RFC text. Newlines are removed whenever 2 or more
        consecutive empty lines are found in the model. This function is a
        part of the model post-processing pipeline.
        :param input_model: The YANG model to be processed
        :return: YANG model with superfluous newlines removed
        """
        ncnt = 0
        output_model = []
        for ln in input_model:
            if ln[0].strip(' \n\r') is '':
                if ncnt is 0:
                    output_model.append(ln)
                elif self.debug_level > 1:
                        self.debug_print_strip_msg(ln[1] - 1, ln[0])
                ncnt += 1
            else:
                output_model.append(ln)
                ncnt = 0
        if self.debug_level > 0:
            print('   Removed %d empty lines' % (len(input_model) - len(output_model)))
        return output_model
def add_line_references(self, input_model):
        """
        This function is a part of the model post-processing pipeline. For
        each line in the module, it adds a reference to the line number in
        the original draft/RFC from where the module line was extracted.
        :param input_model: The YANG model to be processed
        :return: Modified YANG model, where line numbers from the RFC/Draft
                 text file are added as comments at the end of each line in
                 the modified model
        """
        output_model = []
        for ln in input_model:
            line_len = len(ln[0])
            line_ref = ('// %4d' % ln[1]).rjust((self.max_line_len - line_len + 7), ' ')
            new_line = '%s %s\n' % (ln[0].rstrip(' \r\n\t\f'), line_ref)
            output_model.append([new_line, ln[1]])
        return output_model
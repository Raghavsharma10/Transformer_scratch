def strip_empty_lines_backward(self, model, max_lines_to_strip):
        """
        Strips empty lines preceding the line that is currently being parsed. This
        fucntion is called when the parser encounters a Footer.
        :param model: lines that were added to the model up to this point
        :param line_num: the number of teh line being parsed
        :param max_lines_to_strip: max number of lines to strip from the model
        :return: None
        """
        for l in range(0, max_lines_to_strip):
            if model[-1][0].strip(' \r\n\t\f') != '':
                return
            self.debug_print_strip_msg(model[-1][1] - 1, model[-1][0])
            model.pop()
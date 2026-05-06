def post_process_model(self, input_model, add_line_refs):
        """
        This function defines the order and execution logic for actions
        that are performed in the model post-processing pipeline.
        :param input_model: The YANG model to be processed in the pipeline
        :param add_line_refs: Flag that controls whether line number
            references should be added to the model.
        :return: List of strings that constitute the final YANG model to
            be written to its module file.
        """
        intermediate_model = self.remove_leading_spaces(input_model)
        intermediate_model = self.remove_extra_empty_lines(intermediate_model)
        if add_line_refs:
            intermediate_model = self.add_line_references(intermediate_model)
        return finalize_model(intermediate_model)
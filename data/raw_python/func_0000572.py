def check_all(self) -> Generator[AAAError, None, None]:
        """
        Run everything required for checking this function.

        Returns:
            A generator of errors.

        Raises:
            ValidationError: A non-recoverable linting error is found.
        """
        # Function def
        if function_is_noop(self.node):
            return
        self.mark_bl()
        self.mark_def()
        # ACT
        # Load act block and kick out when none is found
        self.act_node = self.load_act_node()
        self.act_block = Block.build_act(self.act_node.node, self.node)
        act_block_first_line_no, act_block_last_line_no = self.act_block.get_span(0)
        # ARRANGE
        self.arrange_block = Block.build_arrange(self.node.body, act_block_first_line_no)
        # ASSERT
        assert self.act_node
        self.assert_block = Block.build_assert(self.node.body, act_block_last_line_no)
        # SPACING
        for block in ['arrange', 'act', 'assert']:
            self_block = getattr(self, '{}_block'.format(block))
            try:
                span = self_block.get_span(self.first_line_no)
            except EmptyBlock:
                continue
            self.line_markers.update(span, self_block.line_type)
        yield from self.line_markers.check_arrange_act_spacing()
        yield from self.line_markers.check_act_assert_spacing()
        yield from self.line_markers.check_blank_lines()
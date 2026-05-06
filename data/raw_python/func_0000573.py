def load_act_node(self) -> ActNode:
        """
        Raises:
            ValidationError: AAA01 when no act block is found and AAA02 when
                multiple act blocks are found.
        """
        act_nodes = ActNode.build_body(self.node.body)

        if not act_nodes:
            raise ValidationError(self.first_line_no, self.node.col_offset, 'AAA01 no Act block found in test')

        # Allow `pytest.raises` and `self.assertRaises()` in assert nodes - if
        # any of the additional nodes are `pytest.raises`, then raise
        for a_n in act_nodes[1:]:
            if a_n.block_type in [ActNodeType.marked_act, ActNodeType.result_assignment]:
                raise ValidationError(
                    self.first_line_no,
                    self.node.col_offset,
                    'AAA02 multiple Act blocks found in test',
                )

        return act_nodes[0]
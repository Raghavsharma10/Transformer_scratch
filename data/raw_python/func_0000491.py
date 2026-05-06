def get_span(self, first_line_no: int) -> Tuple[int, int]:
        """
        Raises:
            EmptyBlock: when block has no nodes
        """
        if not self.nodes:
            raise EmptyBlock('span requested from {} block with no nodes'.format(self.line_type))
        return (
            get_first_token(self.nodes[0]).start[0] - first_line_no,
            get_last_token(self.nodes[-1]).start[0] - first_line_no,
        )
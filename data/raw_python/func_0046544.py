def is_pushdown(self):
        """Tests whether machine is a pushdown automaton."""
        return (self.num_stores == 3 and
                self.state == 0 and self.has_cell(0) and
                self.input == 1 and self.has_input(1) and
                self.has_stack(2))
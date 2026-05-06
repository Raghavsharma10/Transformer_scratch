def is_finite(self):
        """Tests whether machine is a finite automaton."""
        return (self.num_stores == 2 and
                self.state == 0 and self.has_cell(0) and
                self.input == 1 and self.has_input(1))
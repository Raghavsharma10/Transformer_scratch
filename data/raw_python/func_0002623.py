def draw_step(self):
        """iterator that computes all vertices coordinates and edge routing after
           just one step (one layer after the other from top to bottom to top).
           Purely inefficient ! Use it only for "animation" or debugging purpose.
        """
        ostep = self.ordering_step()
        for s in ostep:
            self.setxy()
            self.draw_edges()
            yield s
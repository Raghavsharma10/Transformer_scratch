def latex(self):
        """Gives a latex representation of the assessment."""
        output = self.latex_preamble
        output += self._repr_latex_()
        output += self.latex_post
        return output
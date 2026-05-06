def _repr_latex_(self):
        """
        This is used in IPython notebook it allows us to render the ODEProblem object in LaTeX.
        How Cool is this?
        """
        # TODO: we're mixing HTML with latex here. That is not necessarily a good idea, but works
        # with IPython 1.2.0. Once IPython 2.0 is released, this needs to be changed to _ipython_display_
        lines = []
        lines.append(r"<h1>{0}</h1>".format(self.__class__.__name__))

        lines.append("<p>Method: <code>{0!r}</code></p>".format(self.method))
        lines.append("<p>Parameters: <code>{0!r}</code></p>".format(self.parameters))
        lines.append("<p>Terms:</p>")
        lines.append("<ul>")
        lines.extend(['<li><code>{0!r}</code></li>'.format(lhs) for lhs in self.left_hand_side_descriptors])
        lines.append("</ul>")
        lines.append('<hr />')
        lines.append(r"\begin{align*}")
        for lhs, rhs in zip(self.left_hand_side_descriptors, self.right_hand_side):
            lines.append(r"\dot{{{0}}} &= {1} \\".format(sympy.latex(lhs.symbol), sympy.latex(rhs)))
        lines.append(r"\end{align*}")
        return "\n".join(lines)
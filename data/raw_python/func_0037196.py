def _to_sparky(self):
        """Save :class:`~nmrstarlib.plsimulator.PeakList` into Sparky-formatted string.

        :return: Peak list representation in Sparky format.
        :rtype: :py:class:`str`
        """
        sparky_str = "Assignment\t\t{}\n\n".format("\t\t".join(["w" + str(i + 1) for i in range(len(self.labels))]))
        for peak in self:
            assignment_str = "-".join(peak.assignments_list)
            dimensions_str = "\t\t".join([str(chemshift) for chemshift in peak.chemshifts_list])
            sparky_str += ("{}\t\t{}\n".format(assignment_str, dimensions_str))
        return sparky_str
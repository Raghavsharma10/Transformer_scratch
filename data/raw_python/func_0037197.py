def _to_autoassign(self):
        """Save :class:`~nmrstarlib.plsimulator.PeakList` into AutoAssign-formatted string.

        :return: Peak list representation in AutoAssign format.
        :rtype: :py:class:`str`
        """
        autoassign_str = "#Index\t\t{}\t\tIntensity\t\tWorkbook\n".format(
            "\t\t".join([str(i + 1) + "Dim" for i in range(len(self.labels))]))
        for peak_idx,  peak in enumerate(self):
            dimensions_str = "\t\t".join([str(chemshift) for chemshift in peak.chemshifts_list])
            autoassign_str += "{}\t\t{}\t\t{}\t\t{}\n".format(peak_idx+1, dimensions_str, 0, self.spectrum_name)
        return autoassign_str
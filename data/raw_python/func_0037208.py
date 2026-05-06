def create_peaklist(self, spectrum, chain, chain_idx, source):
        """Create peak list file.

        :param spectrum: Spectrum object instance.
        :type spectrum: :class:`~nmrstarlib.plsimulator.Spectrum`
        :param dict chain: Chain object that contains chemical shift values and assignment information.
        :param int chain_idx: Protein chain index.
        :param str source: :class:`~nmrstarlib.nmrstarlib.StarFile` source.
        :return: Peak list object.
        :rtype: :class:`~nmrstarlib.plsimulator.PeakList`
        """
        sequence_sites = self.create_sequence_sites(chain, spectrum.seq_site_length)
        spin_systems = []
        peaklist = plsimulator.PeakList(spectrum.name, spectrum.labels, source, chain_idx)

        for seq_site in sequence_sites:
            spin_system = plsimulator.SpinSystem()
            for template in spectrum.peak_templates:
                peak = plsimulator.Peak(template.dimension_labels)
                for dim in template:
                    chemshift = seq_site[dim.position].get(dim.label, None)
                    assignment = "{}{}{}".format(seq_site[dim.position]["AA3Code"],
                                                 seq_site[dim.position]["Seq_ID"],
                                                 dim.label)
                    if chemshift and assignment:
                        peak_dim = plsimulator.Dimension(dim.label, dim.position, assignment, float(chemshift))
                        peak.append(peak_dim)
                    else:
                        continue

                if len(peak) == len(template):
                    spin_system.append(peak)
                    peaklist.append(peak)
                else:
                    continue

            spin_systems.append(spin_system)

        if all(len(i) < spectrum.min_spin_system_peaks for i in spin_systems):
            return None

        if self.noise_generator is not None:
            spin_systems_chunks = self.split_by_percent(spin_systems)
            for split_idx, chunk in enumerate(spin_systems_chunks):
                for spin_system in chunk:
                    for peak in spin_system:
                        peak.apply_noise(self.noise_generator, split_idx)

        return peaklist
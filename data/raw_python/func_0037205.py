def create_sequence_sites(chain, seq_site_length):
        """Create sequence sites using sequence ids.

        :param dict chain: Chain object that contains chemical shift values and assignment information.
        :param int seq_site_length: Length of a single sequence site.
        :return: List of sequence sites.
        :rtype: :py:class:`list`
        """
        seq_ids = sorted(list(chain.keys()), key=int)  # make sure that sequence is sorted by sequence id
        slices = [itertools.islice(seq_ids, i, None) for i in range(seq_site_length)]
        seq_site_ids = list(zip(*slices))

        sequence_sites = []
        for seq_site_id in seq_site_ids:
            seq_site = plsimulator.SequenceSite(chain[seq_id] for seq_id in seq_site_id)
            if seq_site.is_sequential():
                sequence_sites.append(seq_site)
            else:
                continue

        return sequence_sites
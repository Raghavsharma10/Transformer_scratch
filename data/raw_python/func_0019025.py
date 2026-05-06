def subs2seqs(self) -> Dict[str, List[str]]:
        """A |collections.defaultdict| containing the node-specific
        information provided by XML `sequences` element.

        >>> from hydpy.auxs.xmltools import XMLInterface
        >>> from hydpy import data
        >>> interface = XMLInterface('single_run.xml', data.get_path('LahnH'))
        >>> series_io = interface.series_io
        >>> subs2seqs = series_io.writers[2].subs2seqs
        >>> for subs, seq in sorted(subs2seqs.items()):
        ...     print(subs, seq)
        node ['sim', 'obs']
        """
        subs2seqs = collections.defaultdict(list)
        nodes = find(self.find('sequences'), 'node')
        if nodes is not None:
            for seq in nodes:
                subs2seqs['node'].append(strip(seq.tag))
        return subs2seqs
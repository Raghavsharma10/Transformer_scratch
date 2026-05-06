def serotype_escherichia(self):
        """
        Create attributes storing the best results for the O and H types
        """
        for sample in self.runmetadata.samples:
            # Initialise negative results to be overwritten when necessary
            sample[self.analysistype].best_o_pid = '-'
            sample[self.analysistype].o_genes = ['-']
            sample[self.analysistype].o_set = ['-']
            sample[self.analysistype].best_h_pid = '-'
            sample[self.analysistype].h_genes = ['-']
            sample[self.analysistype].h_set = ['-']
            if sample.general.bestassemblyfile != 'NA':
                if sample.general.closestrefseqgenus == 'Escherichia':
                    o = dict()
                    h = dict()
                    for result, percentid in sample[self.analysistype].results.items():
                        if 'O' in result.split('_')[-1]:
                            o.update({result: float(percentid)})
                        if 'H' in result.split('_')[-1]:
                            h.update({result: float(percentid)})
                    # O
                    try:
                        sorted_o = sorted(o.items(), key=operator.itemgetter(1), reverse=True)
                        sample[self.analysistype].best_o_pid = str(sorted_o[0][1])

                        sample[self.analysistype].o_genes = [gene for gene, pid in o.items()
                                                             if str(pid) == sample[self.analysistype].best_o_pid]
                        sample[self.analysistype].o_set = \
                            list(set(gene.split('_')[-1] for gene in sample[self.analysistype].o_genes))
                    except (KeyError, IndexError):
                        pass
                    # H
                    try:
                        sorted_h = sorted(h.items(), key=operator.itemgetter(1), reverse=True)
                        sample[self.analysistype].best_h_pid = str(sorted_h[0][1])
                        sample[self.analysistype].h_genes = [gene for gene, pid in h.items()
                                                             if str(pid) == sample[self.analysistype].best_h_pid]
                        sample[self.analysistype].h_set = \
                            list(set(gene.split('_')[-1] for gene in sample[self.analysistype].h_genes))
                    except (KeyError, IndexError):
                        pass
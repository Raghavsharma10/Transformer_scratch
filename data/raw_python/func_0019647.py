def plot_scores(self, motifs, name=True, max_len=50):
        """Create motif scores boxplot of different clusters.
        Motifs can be specified as either motif or factor names.
        The motif scores will be scaled and plotted as z-scores.
        
        Parameters
        ----------
        motifs : iterable or str
            List of motif or factor names.
        
        name : bool, optional
            Use factor names instead of motif names for plotting.
        
        max_len : int, optional
            Truncate the list of factors to this maximum length.
        
        Returns
        -------
        
        g : FacetGrid
            Returns the seaborn FacetGrid object with the plot.
        """
        if self.input.shape[1] != 1:
            raise ValueError("Can't make a categorical plot with real-valued data")
        
        if type("") == type(motifs):
            motifs = [motifs]
            
        plot_motifs = []
        for motif in motifs:
            if motif in self.motifs:
                plot_motifs.append(motif)
            else:
                for m in self.motifs.values():
                    if motif in m.factors:
                        plot_motifs.append(m.id)
        
        data = self.scores[plot_motifs]
        data[:] = data.scale(data, axix=0)
        if name:
            data = data.T
            data["factors"] = [join_max(self.motifs[n].factors, max_len, ",", suffix=",(...)") for n in plot_motifs]
            data = data.set_index("factors").T
        
        data = pd.melt(self.input.join(data), id_vars=["cluster"])
        data.columns = ["cluster", "motif", "z-score"]
        g = sns.factorplot(data=data, y="motif", x="z-score", hue="cluster", kind="box", aspect=2)
        return g
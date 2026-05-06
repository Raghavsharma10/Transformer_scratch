def plot_heatmap(self, kind="final", min_freq=0.01, threshold=2, name=True, max_len=50, aspect=1, **kwargs):
        """Plot clustered heatmap of predicted motif activity.
        
        Parameters
        ----------
        kind : str, optional
            Which data type to use for plotting. Default is 'final', which will plot the 
            result of the rang aggregation. Other options are 'freq' for the motif frequencies,
            or any of the individual activities such as 'rf.score'.
            
        min_freq : float, optional
            Minimum frequency of motif occurrence.
            
        threshold : float, optional
            Minimum activity (absolute) of the rank aggregation result. 
        
        name : bool, optional
            Use factor names instead of motif names for plotting.
        
        max_len : int, optional
            Truncate the list of factors to this maximum length.
            
        aspect : int, optional
            Aspect ratio for tweaking the plot.
            
        kwargs : other keyword arguments
            All other keyword arguments are passed to sns.clustermap

        Returns
        -------
        cg : ClusterGrid
            A seaborn ClusterGrid instance.
        """
        
        filt = np.any(np.abs(self.result) >= threshold, 1) & np.any(np.abs(self.freq.T) >= min_freq, 1)
        
        idx = self.result[filt].index
        
        cmap = "RdBu_r" 
        if kind == "final":
            data = self.result
        elif kind == "freq":
            data = self.freq.T
            cmap = "Reds"
        elif kind in self.activity:
            data = self.activity[dtype]
            if kind in ["hypergeom.count", "mwu.score"]:
                cmap = "Reds"
        else:
            raise ValueError("Unknown dtype")
        
        #print(data.head())
        #plt.figure(
        m = data.loc[idx]
        if name:
            m["factors"] = [join_max(self.motifs[n].factors, max_len, ",", suffix=",(...)") for n in m.index]
            m = m.set_index("factors")
        h,w = m.shape
        cg = sns.clustermap(m, cmap=cmap, col_cluster=False, 
                            figsize=(2 + w * 0.5 * aspect, 0.5 * h), linewidths=1,
                           **kwargs)
        cg.ax_col_dendrogram.set_visible(False)
        plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0);
        return cg
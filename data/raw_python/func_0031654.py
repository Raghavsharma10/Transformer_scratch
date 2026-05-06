def plotstuff(self, T=[0, 1000]):
        """
        Create a scatter plot of the contents of the database,
        with entries on the interval T.


        Parameters
        ----------
        T : list
            Time interval.
        
        
        Returns
        -------
        None
        
        
        See also
        --------
        GDF.select_neurons_interval
        """

        fig = plt.figure(figsize=(10,10))

        ax = fig.add_subplot(111)

        neurons = self.neurons()
        i = 0
        for x in self.select_neurons_interval(neurons, T):
            ax.plot(x, np.zeros(x.size) + neurons[i], 'o',
                    markersize=1, markerfacecolor='k', markeredgecolor='k',
                    alpha=0.25)
            i += 1
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('neuron ID')
        ax.set_xlim(T[0], T[1])
        ax.set_ylim(neurons.min(), neurons.max())
        ax.set_title('database content on T = [%.0f, %.0f]' % (T[0], T[1]))
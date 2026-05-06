def add_sst_to_dot_display(ax, sst, color= '0.',alpha= 1.):
    '''
    suitable for plotting fraction of neurons
    '''
    plt.sca(ax)
    N = len(sst)
    current_ymax = 0
    counter = 0
    while True:
        if len(ax.get_lines()) !=0:
            data = ax.get_lines()[-1-counter].get_data()[1]
            if np.sum(data) != 0: # if not empty array 
                current_ymax = np.max(data)
                break
            counter +=1
        else: 
            break
    for i in np.arange(N):
        plt.plot(sst[i],np.ones_like(sst[i])+i+current_ymax -1, 'k o',ms=0.5, mfc=color,mec=color, alpha=alpha)
    plt.xlabel(r'time (ms)')
    plt.ylabel(r'neuron id')
    return ax
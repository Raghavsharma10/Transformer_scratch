def plot_population(ax,
                    populationParams,
                    electrodeParams,
                    layerBoundaries,
                    aspect='equal',
                    isometricangle=np.pi/12,
                    X=['EX', 'IN'],
                    markers=['^', 'o'],
                    colors=['r', 'b'],
                    layers = ['upper', 'lower'],
                    title='positions'):
    '''
    Plot the geometry of the column model, optionally with somatic locations
    and optionally with reconstructed neurons
    
    kwargs:
    ::
        ax : matplotlib.axes.AxesSubplot
        aspect : str
            matplotlib.axis argument
        isometricangle : float
            pseudo-3d view angle
        plot_somas : bool
            plot soma locations
        plot_morphos : bool
            plot full morphologies
        num_unitsE : int
            number of excitatory morphos plotted per population
        num_unitsI : int
            number of inhibitory morphos plotted per population
        clip_dendrites : bool
            draw dendrites outside of axis
        mainpops : bool
            if True, plot only main pops, e.g. b23 and nb23 as L23I
    
    return:
    ::
        axis : list
            the plt.axis() corresponding to input aspect
    '''

    remove_axis_junk(ax, ['right', 'bottom', 'left', 'top'])

    
    # DRAW OUTLINE OF POPULATIONS 
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    #contact points
    ax.plot(electrodeParams['x'],
            electrodeParams['z'],
            '.', marker='o', markersize=5, color='k', zorder=0)

    #outline of electrode       
    x_0 = np.array(electrodeParams['r_z'])[1, 1:-1]
    z_0 = np.array(electrodeParams['r_z'])[0, 1:-1]
    x = np.r_[x_0[-1], x_0[::-1], -x_0[1:], -x_0[-1]]
    z = np.r_[100, z_0[::-1], z_0[1:], 100]
    ax.fill(x, z, color=(0.5, 0.5, 0.5), lw=None, zorder=-0.1)

    #outline of populations:
    #fetch the population radius from some population
    r = populationParams[populationParams.keys()[0]]['radius']

    theta0 = np.linspace(0, np.pi, 20)
    theta1 = np.linspace(np.pi, 2*np.pi, 20)
    
    zpos = np.r_[np.array(layerBoundaries)[:, 0],
                 np.array(layerBoundaries)[-1, 1]]
    
    for i, z in enumerate(np.mean(layerBoundaries, axis=1)):
        ax.text(r, z, ' %s' % layers[i],
                va='center', ha='left', rotation='vertical')

    for i, zval in enumerate(zpos):
        if i == 0:
            ax.plot(r*np.cos(theta0),
                    r*np.sin(theta0)*np.sin(isometricangle)+zval,
                    color='k', zorder=-r, clip_on=False)
            ax.plot(r*np.cos(theta1),
                    r*np.sin(theta1)*np.sin(isometricangle)+zval,
                    color='k', zorder=r, clip_on=False)
        else:
            ax.plot(r*np.cos(theta0),
                    r*np.sin(theta0)*np.sin(isometricangle)+zval,
                    color='gray', zorder=-r, clip_on=False)
            ax.plot(r*np.cos(theta1),
                    r*np.sin(theta1)*np.sin(isometricangle)+zval,
                    color='k', zorder=r, clip_on=False)
    
    ax.plot([-r, -r], [zpos[0], zpos[-1]], 'k', zorder=0, clip_on=False)
    ax.plot([r, r], [zpos[0], zpos[-1]], 'k', zorder=0, clip_on=False)
    
    #plot a horizontal radius scalebar
    ax.plot([0, r], [z_0.min()]*2, 'k', lw=2, zorder=0, clip_on=False)
    ax.text(r / 2., z_0.min()-100, 'r = %i $\mu$m' % int(r), ha='center')
    
    #plot a vertical depth scalebar
    ax.plot([-r]*2, [z_0.min()+50, z_0.min()-50],
        'k', lw=2, zorder=0, clip_on=False)
    ax.text(-r, z_0.min(), r'100 $\mu$m', va='center', ha='right')
    
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    #fake ticks:
    for pos in zpos:
        ax.text(-r, pos, 'z=%i-' % int(pos), ha='right', va='center')
 
    ax.set_title(title)
   
    axis = ax.axis(ax.axis(aspect))
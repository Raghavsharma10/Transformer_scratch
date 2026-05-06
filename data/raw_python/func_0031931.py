def plot_multi_scale_output_b(fig, X='L5E'):
    '''docstring me'''

    show_ax_labels = True
    show_insets = False
    show_images = False

    T=[800, 1000]
    T_inset=[900, 920]

    
    left = 0.075
    bottom = 0.05
    top = 0.475
    right = 0.95
    axwidth = 0.16
    numcols = 4
    insetwidth = axwidth/2
    insetheight = 0.5
    
    lefts = np.linspace(left, right-axwidth, numcols)
    lefts += axwidth/2



    #lower row of panels
    #fig = plt.figure()
    #fig.subplots_adjust(left=0.12, right=0.9, bottom=0.36, top=0.9, wspace=0.2, hspace=0.3)

    ############################################################################    
    # E part, soma locations
    ############################################################################

    ax4 = fig.add_axes([lefts[0], bottom, axwidth, top-bottom], frameon=False)
    plt.locator_params(nbins=4)
    ax4.xaxis.set_ticks([])
    ax4.yaxis.set_ticks([])
    if show_ax_labels:
        phlp.annotate_subplot(ax4, ncols=4, nrows=1, letter='E')
    plot_population(ax4, params, isometricangle=np.pi/24, rasterized=False)
    
    
    ############################################################################    
    # F part, CSD
    ############################################################################

    ax5 = fig.add_axes([lefts[1], bottom, axwidth, top-bottom])
    plt.locator_params(nbins=4)
    phlp.remove_axis_junk(ax5)
    if show_ax_labels:
        phlp.annotate_subplot(ax5, ncols=4, nrows=1, letter='F')
    plot_signal_sum(ax5, params, fname=os.path.join(params.savefolder, 'CSDsum.h5'),
                        unit='$\mu$A mm$^{-3}$',
                        T=T,
                        ylim=[ax4.axis()[2], ax4.axis()[3]],
                        rasterized=False)
    ax5.set_title('CSD', va='center')
    
    # Inset
    if show_insets:
        ax6 = fig.add_axes([lefts[1]+axwidth-insetwidth, top-insetheight, insetwidth, insetheight])
        plt.locator_params(nbins=4)
        phlp.remove_axis_junk(ax6)
        plot_signal_sum_colorplot(ax6, params, os.path.join(params.savefolder, 'CSDsum.h5'),
                                  unit=r'$\mu$Amm$^{-3}$', T=T_inset,
                                  ylim=[ax4.axis()[2], ax4.axis()[3]],
                                  fancy=False,colorbar=False,cmap='bwr_r')
        ax6.set_xticks(T_inset)
        ax6.set_yticklabels([])

    #show traces superimposed on color image
    if show_images:
        plot_signal_sum_colorplot(ax5, params, os.path.join(params.savefolder, 'CSDsum.h5'),
                                  unit=r'$\mu$Amm$^{-3}$', T=T,
                                  ylim=[ax4.axis()[2], ax4.axis()[3]],
                                  fancy=False,colorbar=False,cmap='jet_r')
        

    
    ############################################################################
    # G part, LFP 
    ############################################################################

    ax7 = fig.add_axes([lefts[2], bottom, axwidth, top-bottom])
    plt.locator_params(nbins=4)
    if show_ax_labels:
        phlp.annotate_subplot(ax7, ncols=4, nrows=1, letter='G')
    phlp.remove_axis_junk(ax7)
    plot_signal_sum(ax7, params, fname=os.path.join(params.savefolder, 'LFPsum.h5'),
                    unit='mV', T=T, ylim=[ax4.axis()[2], ax4.axis()[3]],
                    rasterized=False)
    ax7.set_title('LFP',va='center')
    
    # Inset
    if show_insets:
        ax8 = fig.add_axes([lefts[2]+axwidth-insetwidth, top-insetheight, insetwidth, insetheight])
        plt.locator_params(nbins=4)
        phlp.remove_axis_junk(ax8)
        plot_signal_sum_colorplot(ax8, params, os.path.join(params.savefolder, 'LFPsum.h5'),
                                  unit='mV', T=T_inset,
                                  ylim=[ax4.axis()[2], ax4.axis()[3]],
                                  fancy=False,colorbar=False,cmap='bwr_r')   
        ax8.set_xticks(T_inset)
        ax8.set_yticklabels([])

    #show traces superimposed on color image
    if show_images:
        plot_signal_sum_colorplot(ax7, params, os.path.join(params.savefolder, 'LFPsum.h5'),
                                  unit='mV', T=T,
                                  ylim=[ax4.axis()[2], ax4.axis()[3]],
                                  fancy=False,colorbar=False,cmap='bwr_r')
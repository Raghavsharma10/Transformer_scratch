def plotOptMod(verNObg3gray, VERgray):
    """ called from either readTranscar.py or hist-feasibility/plotsnew.py """
    if VERgray is None and verNObg3gray is None:
        return

    fg = figure()
    ax2 = fg.gca()  # summed (as camera would see)

    if VERgray is not None:
        z = VERgray.alt_km
        Ek = VERgray.energy_ev.values

#        ax1.semilogx(VERgray, z, marker='',label='filt', color='b')
        props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.5}
        fgs, axs = fg.subplots(6, 6, sharex=True, sharey='row')
        axs = axs.ravel()  # for convenient iteration
        fgs.subplots_adjust(hspace=0, wspace=0)
        fgs.suptitle('filtered VER/flux')
        fgs.text(0.04, 0.5, 'Altitude [km]', va='center', rotation='vertical')
        fgs.text(0.5, 0.04, 'Beam energy [eV]', ha='center')
        for i, e in enumerate(Ek):
            axs[i].semilogx(VERgray.loc[:, e], z)
            axs[i].set_xlim((1e-3, 1e4))

# place a text box in upper left in axes coords
            axs[i].text(0.95, 0.95, '{:0.0f}'.format(e)+'eV',
                        transform=axs[i].transAxes, fontsize=12,
                        va='top', ha='right', bbox=props)
        for i in range(33, 36):
            axs[i].axis('off')

        ax2.semilogx(VERgray.sum(axis=1), z, label='filt', color='b')

        # specific to energies
        ax = figure().gca()
        for e in Ek:
            ax.semilogx(VERgray.loc[:, e], z, marker='', label='{:.0f} eV'.format(e))
        ax.set_title('filtered VER/flux')
        ax.set_xlabel('VER/flux')
        ax.set_ylabel('altitude [km]')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim((1e-5, 1e5))
        ax.grid(True)

    if verNObg3gray is not None:
        ax1 = figure().gca()  # overview
        z = verNObg3gray.alt_km
        Ek = verNObg3gray.energy_ev.values

        ax1.semilogx(verNObg3gray, z, marker='', label='unfilt', color='r')
        ax2.semilogx(verNObg3gray.sum(axis=1), z, label='unfilt', color='r')

        ax = figure().gca()
        for e in Ek:
            ax.semilogx(verNObg3gray.loc[:, e], z, marker='', label='{:.0f} eV'.format(e))
        ax.set_title('UNfiltered VER/flux')
        ax.set_xlabel('VER/flux')
        ax.set_ylabel('altitude [km]')
        ax.legend(loc='best', fontsize=8)
        ax.set_xlim((1e-5, 1e5))
        ax.grid(True)

        ax1.set_title('VER/flux, one profile per beam')
        ax1.set_xlabel('VER/flux')
        ax1.set_ylabel('altitude [km]')
        ax1.grid(True)

    ax2.set_xlabel('VER/flux')
    ax2.set_ylabel('altitude [km]')
    ax2.set_title('VER/flux summed over all energy beams \n (as the camera would see)')
    ax2.legend(loc='best')
    ax2.grid(True)
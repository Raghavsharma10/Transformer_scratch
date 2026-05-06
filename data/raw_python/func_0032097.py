def K2onSilicon(infile, fieldnum, do_nearSiliconCheck=False):
    """Checks whether targets are on silicon during a given campaign.

    This function will write a csv table called targets_siliconFlag.csv,
    which details the silicon status for each target listed in `infile`
    (0 = not on silicon, 2 = on silion).

    Parameters
    ----------
    infile : str
        Path to a csv table with columns ra_deg,dec_deg,magnitude (no header).

    fieldnum : int
        K2 Campaign number.

    do_nearSiliconCheck : bool
        If `True`, targets near (but not on) silicon are flagged with a "1".
    """
    ra_sources_deg, dec_sources_deg, mag = parse_file(infile)
    n_sources = np.shape(ra_sources_deg)[0]
    if n_sources > 500:
        logger.warning("Warning: there are {0} sources in your target list, "
                       "this could take some time".format(n_sources))

    k = fields.getKeplerFov(fieldnum)
    raDec = k.getCoordsOfChannelCorners()

    onSilicon = list(
                    map(
                        onSiliconCheck,
                        ra_sources_deg,
                        dec_sources_deg,
                        np.repeat(k, len(ra_sources_deg))
                        )
                    )
    onSilicon = np.array(onSilicon, dtype=bool)

    if do_nearSiliconCheck:
        nearSilicon = list(
                        map(
                            nearSiliconCheck,
                            ra_sources_deg,
                            dec_sources_deg,
                            np.repeat(k, len(ra_sources_deg))
                            )
                        )
        nearSilicon = np.array(nearSilicon, dtype=bool)

    if got_mpl:
        almost_black = '#262626'
        light_grey = np.array([float(248)/float(255)]*3)
        ph = proj.PlateCaree()
        k.plotPointing(ph, showOuts=False)
        targets = ph.skyToPix(ra_sources_deg, dec_sources_deg)
        targets = np.array(targets)
        fig = pl.gcf()
        ax = fig.gca()
        ax = fig.add_subplot(111)
        ax.scatter(*targets, color='#fc8d62', s=7, label='not on silicon')
        ax.scatter(targets[0][onSilicon], targets[1][onSilicon],
                   color='#66c2a5', s=8, label='on silicon')
        ax.set_xlabel('R.A. [degrees]', fontsize=16)
        ax.set_ylabel('Declination [degrees]', fontsize=16)
        ax.invert_xaxis()
        ax.minorticks_on()
        legend = ax.legend(loc=0, frameon=True, scatterpoints=1)
        rect = legend.get_frame()
        rect.set_alpha(0.3)
        rect.set_facecolor(light_grey)
        rect.set_linewidth(0.0)
        texts = legend.texts
        for t in texts:
            t.set_color(almost_black)
        fig.savefig('targets_fov.png', dpi=300)
        pl.close('all')

    # prints zero if target is not on silicon
    siliconFlag = np.zeros_like(ra_sources_deg)

    # print a 1 if target is near but not on silicon
    if do_nearSiliconCheck:
        siliconFlag = np.where(nearSilicon, 1, siliconFlag)

    # prints a 2 if target is on silicon
    siliconFlag = np.where(onSilicon, 2, siliconFlag)

    outarr = np.array([ra_sources_deg, dec_sources_deg, mag, siliconFlag])
    np.savetxt('targets_siliconFlag.csv', outarr.T, delimiter=', ',
               fmt=['%10.10f', '%10.10f', '%10.2f', '%i'])

    if got_mpl:
        print('I made two files: targets_siliconFlag.csv and targets_fov.png')
    else:
        print('I made one file: targets_siliconFlag.csv')
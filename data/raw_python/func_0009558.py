def plotSet(imgDir, posExTime, outDir, show_legend,
            show_plots, save_to_file, ftype):
    '''
    creates plots showing both found GAUSSIAN peaks, the histogram, a smoothed histogram
    from all images within [imgDir]

    posExTime - position range of the exposure time in the image name e.g.: img_30s.jpg -> (4,5)
    outDir - dirname to save the output images
    show_legend - True/False
    show_plots - display the result on screen
    save_to_file - save the result to file
    ftype - file type of the output images
    '''
    xvals = []
    hist = []
    peaks = []
    exTimes = []
    max_border = 0

    if not imgDir.exists():
        raise Exception("image dir doesn't exist")

    for n, f in enumerate(imgDir):
        print(f)
        try:
            # if imgDir.join(f).isfile():
            img = imgDir.join(f)
            s = FitHistogramPeaks(img)
            xvals.append(s.xvals)
            hist.append(s.yvals)
#             smoothedHist.append(s.yvals2)
            peaks.append(s.fitValues())

            if s.border() > max_border:
                max_border = s.plotBorder()

            exTimes.append(float(f[posExTime[0]:posExTime[1] + 1]))
        except:
            pass
    nx = 2
    ny = int(len(hist) // nx) + len(hist) % nx

    fig, ax = plt.subplots(ny, nx)

    # flatten 2d-ax list:
    if nx > 1:
        ax = [list(i) for i in zip(*ax)]  # transpose 2d-list
        axx = []
        for xa in ax:
            for ya in xa:
                axx.append(ya)
        ax = axx

    for x, h, p, e, a in zip(xvals, hist, peaks, exTimes, ax):

        a.plot(x, h, label='histogram', thickness=3)
#         l1 = a.plot(x, s, label='smoothed')
        for n, pi in enumerate(p):
            l2 = a.plot(x, pi, label='peak %s' % n, thickness=6)
        a.set_xlim(xmin=0, xmax=max_border)
        a.set_title('%s s' % e)

# plt.setp([l1,l2], linewidth=2)#, linestyle='--', color='r')       # set
# both to dashed

    l1 = ax[0].legend()  # loc='upper center', bbox_to_anchor=(0.7, 1.05),
    l1.draw_frame(False)

    plt.xlabel('pixel value')
    plt.ylabel('number of pixels')

    fig = plt.gcf()
    fig.set_size_inches(7 * nx, 3 * ny)

    if save_to_file:
        p = PathStr(outDir).join('result').setFiletype(ftype)
        plt.savefig(p, bbox_inches='tight')

    if show_plots:
        plt.show()
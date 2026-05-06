def make_layout(maxval):
    """Make the physical layout of the MinION flowcell.
    based on https://bioinformatics.stackexchange.com/a/749/681
    returned as a numpy array
    """
    if maxval > 512:
        return Layout(
            structure=np.concatenate([np.array([list(range(10 * i + 1, i * 10 + 11))
                                                for i in range(25)]) + j
                                      for j in range(0, 3000, 250)],
                                     axis=1),
            template=np.zeros((25, 120)),
            xticks=range(1, 121),
            yticks=range(1, 26))
    else:
        layoutlist = []
        for i, j in zip(
                [33, 481, 417, 353, 289, 225, 161, 97],
                [8, 456, 392, 328, 264, 200, 136, 72]):
            for n in range(4):
                layoutlist.append(list(range(i + n * 8, (i + n * 8) + 8, 1)) +
                                  list(range(j + n * 8, (j + n * 8) - 8, -1)))
        return Layout(
            structure=np.array(layoutlist).transpose(),
            template=np.zeros((16, 32)),
            xticks=range(1, 33),
            yticks=range(1, 17))
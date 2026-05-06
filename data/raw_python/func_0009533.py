def removeSinglePixels(img):
    '''
    img - boolean array
    remove all pixels that have no neighbour
    '''

    gx = img.shape[0]
    gy = img.shape[1]

    for i in range(gx):
        for j in range(gy):

            if img[i, j]:

                found_neighbour = False
                for ii in range(max(0, i - 1), min(gx, i + 2)):
                    for jj in range(max(0, j - 1), min(gy, j + 2)):

                        if ii == i and jj == j:
                            continue

                        if img[ii, jj]:
                            found_neighbour = True
                            break
                    if found_neighbour:
                        break

                if not found_neighbour:
                    img[i, j] = 0
def get_colors(num=16, cmap=plt.cm.Set1):
    '''return a list of color tuples to use in plots'''
    colors = []
    for i in xrange(num):
        if analysis_params.bw:
            colors.append('k' if i % 2 == 0 else 'gray')
        else:
            i *= 256.
            if num > 1:
                i /= num - 1.
            else:
                i /= num
            colors.append(cmap(int(i)))
    return colors
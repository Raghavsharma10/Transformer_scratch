def normalize(x):
    '''normalize x to have mean 0 and unity standard deviation'''
    x = x.astype(float)
    x -= x.mean()
    return x / float(x.std())
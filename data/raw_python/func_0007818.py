def plot(hdiff, title):
    """ Plots the tropical solar length
    by year.
    
    """
    import matplotlib.pyplot as plt
    years = [elem[0] for elem in hdiff]
    diffs = [elem[1] for elem in hdiff]
    plt.plot(years, diffs)
    plt.ylabel('Distance in minutes')
    plt.xlabel('Year')
    plt.title(title)
    plt.axhline(y=0, c='red')
    plt.show()
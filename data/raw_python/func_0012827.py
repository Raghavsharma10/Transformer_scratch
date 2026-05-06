def printcsv(csvdiffs):
    """print the csv"""
    for row in csvdiffs:
        print(','.join([str(cell) for cell in row]))
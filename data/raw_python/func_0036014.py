def csv(file, *args, **kwargs):
    '''
    Write CSV file.

    Parameters
    ----------
    file : Path
    *args
        csv.DictWriter args (except the f arg)
    **kwargs
        csv.DictWriter args

    Examples
    --------
    with write.csv(file) as writer:
        writer.writerow((1,2,3))
    '''
    with file.open('w', newline='') as f:
        yield DictWriter(f, *args, **kwargs)
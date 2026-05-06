def pbar_strings(files, desc='', **kwargs):
    """Wrapper for `tqdm` progress bar which also sorts list of strings
    """
    return tqdm(
        sorted(files, key=lambda s: s.lower()),
        desc=('<' + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '> ' +
              desc),
        dynamic_ncols=True,
        **kwargs)
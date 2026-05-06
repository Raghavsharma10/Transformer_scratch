def pbar(iter, desc='', **kwargs):
    """Wrapper for `tqdm` progress bar.
    """
    return tqdm(
        iter,
        desc=('<' + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + '> ' +
              desc),
        dynamic_ncols=True,
        **kwargs)
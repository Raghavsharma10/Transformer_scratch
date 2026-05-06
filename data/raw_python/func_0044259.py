def get(ctx, dataset, rm, keep_compressed, dont_process, dont_download, keep_raw, kwargs):
    "performs the operations download, extract, rm_compressed, processes and rm_raw, in sequence. KWARGS must be in the form: key=value, and are fowarded to all opeartions."

    kwargs = parse_kwargs(kwargs)

    process = not dont_process
    rm_raw = not keep_raw
    rm_compressed = not keep_compressed
    download = not dont_download

    data(dataset, **ctx.obj).get(download=download, rm=rm, rm_compressed=rm_compressed, process=process, rm_raw=rm_raw, **kwargs)
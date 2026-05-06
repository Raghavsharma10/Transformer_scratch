def ffmpeg(*args, **kwargs):
    """Universal ffmpeg wrapper with progress and error handling"""

    ff = FFMPEG(*args, **kwargs)
    ff.start(
            stdin=kwargs.get("stdin", None),
            stdout=kwargs.get("stdout", None),
            stderr=kwargs.get("stderr", subprocess.PIPE)
        )

    ff.wait(kwargs.get("progress_handler", None))

    if ff.return_code:
        err = indent(ff.error_log)
        logging.error("Problem occured during transcoding\n\n{}\n\n".format(err))
        return False
    return True
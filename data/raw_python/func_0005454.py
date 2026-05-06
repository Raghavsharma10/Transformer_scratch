def write_file(path, content, mode='w'):
    # type: (Text, Union[Text,bytes], Text) -> None
    """ --pretend aware file writing.

    You can always write files manually but you should always handle the
    --pretend case.

    Args:
        path (str):
        content (str):
        mode (str):
    """
    from peltak.core import context
    from peltak.core import log

    if context.get('pretend', False):
        log.info("Would overwrite <34>{path}<32> with:\n<90>{content}",
                 path=path,
                 content=content)
    else:
        with open(path, mode) as fp:
            fp.write(content)
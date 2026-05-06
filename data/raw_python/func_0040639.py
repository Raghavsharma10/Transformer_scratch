def get_frame_src(f:Frame) -> str:
    ''' inspects a frame and returns a string with the following

        <src-path>:<src-line> -> <function-name>
        <source-code>
    '''
    path, line, src, fn = _get_frame(
        inspect.getframeinfo(f)
    )
    return '{}:{} -> {}\n{}'.format(
        path.split(os.sep)[-1],
        line,
        fn,
        repr(src[0][:-1]) # shave off \n
    )
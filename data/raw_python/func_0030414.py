def patch_file_open(): # pragma: no cover
    """A Monkey patch to log opening and closing of files, which is useful for
    debugging file descriptor exhaustion."""

    openfiles = set()
    oldfile = builtins.file

    class newfile(oldfile):
        def __init__(self, *args, **kwargs):
            self.x = args[0]

            all_fds = count_open_fds()

            print('### {} OPENING {} ( {} total )###'.format(
                len(openfiles), str(self.x), all_fds))
            oldfile.__init__(self, *args, **kwargs)

            openfiles.add(self)

        def close(self):
            print('### {} CLOSING {} ###'.format(len(openfiles), str(self.x)))
            oldfile.close(self)
            openfiles.remove(self)

    def newopen(*args, **kwargs):
        return newfile(*args, **kwargs)

    builtins.file = newfile
    builtins.open = newopen
def _progress_hook(self, blocknum, blocksize, totalsize):
        """ Progress hook for urlretrieve. """
        read = blocknum * blocksize
        if totalsize > 0:
            percent = read * 1e2 / totalsize
            s = "\r%d%% %*d / %d" % (
                percent, len(str(totalsize)), read, totalsize)
            sys.stdout.write(s)

            if read >= totalsize:
                sys.stdout.write("\n")
        else:
            sys.stdout.write("read %d\n" % read)
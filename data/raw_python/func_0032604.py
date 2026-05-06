def bunzip2(filename):
    """Uncompress `filename` in place"""
    log.debug("Uncompressing %s", filename)
    tmpfile = "%s.tmp" % filename
    os.rename(filename, tmpfile)
    b = bz2.BZ2File(tmpfile)
    f = open(filename, "wb")
    while True:
        block = b.read(512 * 1024)
        if not block:
            break
        f.write(block)
    f.close()
    b.close()
    shutil.copystat(tmpfile, filename)
    shutil.copymode(tmpfile, filename)
    os.unlink(tmpfile)
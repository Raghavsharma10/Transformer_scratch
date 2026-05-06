def _open_next(self):
        """Proceed to next volume."""

        # is the file split over archives?
        if (self._cur.flags & rarfile.RAR_FILE_SPLIT_AFTER) == 0:
            return False

        if self._fd:
            self._fd.close()
            self._fd = None

        # open next part
        self._volfile = self._parser._next_volname(self._volfile)
        fd = rarfile.XFile(self._volfile)
        self._fd = fd
        sig = fd.read(len(self._parser._expect_sig))
        if sig != self._parser._expect_sig:
            raise rarfile.BadRarFile("Invalid signature")

        # loop until first file header
        while 1:
            cur = self._parser._parse_header(fd)
            if not cur:
                raise rarfile.BadRarFile("Unexpected EOF")
            if cur.type in (rarfile.RAR_BLOCK_MARK, rarfile.RAR_BLOCK_MAIN):
                if cur.add_size:
                    fd.seek(cur.add_size, 1)
                continue
            if cur.orig_filename != self._inf.orig_filename:
                raise rarfile.BadRarFile("Did not found file entry")
            self._cur = cur
            self._cur_avail = cur.add_size
            return True
def _handle_bom(self, infile):
        """
        Handle any BOM, and decode if necessary.

        If an encoding is specified, that *must* be used - but the BOM should
        still be removed (and the BOM attribute set).

        (If the encoding is wrongly specified, then a BOM for an alternative
        encoding won't be discovered or removed.)

        If an encoding is not specified, UTF8 or UTF16 BOM will be detected and
        removed. The BOM attribute will be set. UTF16 will be decoded to
        unicode.

        NOTE: This method must not be called with an empty ``infile``.

        Specifying the *wrong* encoding is likely to cause a
        ``UnicodeDecodeError``.

        ``infile`` must always be returned as a list of lines, but may be
        passed in as a single string.
        """
        if ((self.encoding is not None) and
            (self.encoding.lower() not in BOM_LIST)):
            # No need to check for a BOM
            # the encoding specified doesn't have one
            # just decode
            return self._decode(infile, self.encoding)

        if isinstance(infile, (list, tuple)):
            line = infile[0]
        else:
            line = infile
        if self.encoding is not None:
            # encoding explicitly supplied
            # And it could have an associated BOM
            # TODO: if encoding is just UTF16 - we ought to check for both
            # TODO: big endian and little endian versions.
            enc = BOM_LIST[self.encoding.lower()]
            if enc == 'utf_16':
                # For UTF16 we try big endian and little endian
                for BOM, (encoding, final_encoding) in list(BOMS.items()):
                    if not final_encoding:
                        # skip UTF8
                        continue
                    if infile.startswith(BOM):
                        ### BOM discovered
                        ##self.BOM = True
                        # Don't need to remove BOM
                        return self._decode(infile, encoding)

                # If we get this far, will *probably* raise a DecodeError
                # As it doesn't appear to start with a BOM
                return self._decode(infile, self.encoding)

            # Must be UTF8
            BOM = BOM_SET[enc]
            if not line.startswith(BOM):
                return self._decode(infile, self.encoding)

            newline = line[len(BOM):]

            # BOM removed
            if isinstance(infile, (list, tuple)):
                infile[0] = newline
            else:
                infile = newline
            self.BOM = True
            return self._decode(infile, self.encoding)

        # No encoding specified - so we need to check for UTF8/UTF16
        for BOM, (encoding, final_encoding) in list(BOMS.items()):
            if not isinstance(BOM, str) or not line.startswith(BOM):
                continue
            else:
                # BOM discovered
                self.encoding = final_encoding
                if not final_encoding:
                    self.BOM = True
                    # UTF8
                    # remove BOM
                    newline = line[len(BOM):]
                    if isinstance(infile, (list, tuple)):
                        infile[0] = newline
                    else:
                        infile = newline
                    # UTF8 - don't decode
                    if isinstance(infile, string_types):
                        return infile.splitlines(True)
                    else:
                        return infile
                # UTF16 - have to decode
                return self._decode(infile, encoding)

        # No BOM discovered and no encoding specified, just return
        if isinstance(infile, string_types):
            # infile read from a file will be a single string
            return infile.splitlines(True)
        return infile
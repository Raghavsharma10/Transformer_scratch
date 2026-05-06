def get_key(bytes_, encoding, keynames='curtsies', full=False):
    """Return key pressed from bytes_ or None

    Return a key name or None meaning it's an incomplete sequence of bytes
    (more bytes needed to determine the key pressed)

    encoding is how the bytes should be translated to unicode - it should
    match the terminal encoding.

    keynames is a string describing how keys should be named:

    * curtsies uses unicode strings like <F8>

    * curses uses unicode strings similar to those returned by
      the Python ncurses window.getkey function, like KEY_F(8),
      plus a nonstandard representation of meta keys (bytes 128-255)
      because returning the corresponding unicode code point would be
      indistinguishable from the multibyte sequence that encodes that
      character in the current encoding

    * bytes returns the original bytes from stdin (NOT unicode)

    if full, match a key even if it could be a prefix to another key
    (useful for detecting a plain escape key for instance, since
    escape is also a prefix to a bunch of char sequences for other keys)

    Events are subclasses of Event, or unicode strings

    Precondition: get_key(prefix, keynames) is None for all proper prefixes of
    bytes. This means get_key should be called on progressively larger inputs
    (for 'asdf', first on 'a', then on 'as', then on 'asd' - until a non-None
    value is returned)
    """
    if not all(isinstance(c, type(b'')) for c in bytes_):
        raise ValueError("get key expects bytes, got %r" % bytes_) # expects raw bytes
    if keynames not in ['curtsies', 'curses', 'bytes']:
        raise ValueError("keynames must be one of 'curtsies', 'curses' or 'bytes'")
    seq = b''.join(bytes_)
    if len(seq) > MAX_KEYPRESS_SIZE:
        raise ValueError('unable to decode bytes %r' % seq)

    def key_name():
        if keynames == 'curses':
            if seq in CURSES_NAMES: # may not be here (and still not decodable) curses names incomplete
                return CURSES_NAMES[seq]

            # Otherwise, there's no special curses name for this
            try:
                return seq.decode(encoding) # for normal decodable text or a special curtsies sequence with bytes that can be decoded
            except UnicodeDecodeError:
                # this sequence can't be decoded with this encoding, so we need to represent the bytes
                if len(seq) == 1:
                    return u'x%02X' % ord(seq)
                    #TODO figure out a better thing to return here
                else:
                    raise NotImplementedError("are multibyte unnameable sequences possible?")
                    return u'bytes: ' + u'-'.join(u'x%02X' % ord(seq[i:i+1]) for i in range(len(seq)))
                    #TODO if this isn't possible, return multiple meta keys as a paste event if paste events enabled
        elif keynames == 'curtsies':
            if seq in CURTSIES_NAMES:
                return CURTSIES_NAMES[seq]
            return seq.decode(encoding) #assumes that curtsies names are a subset of curses ones
        else:
            assert keynames == 'bytes'
            return seq

    key_known = seq in CURTSIES_NAMES or seq in CURSES_NAMES or decodable(seq, encoding)

    if full and key_known:
        return key_name()
    elif seq in KEYMAP_PREFIXES or could_be_unfinished_char(seq, encoding):
        return None # need more input to make up a full keypress
    elif key_known:
        return key_name()
    else:
        seq.decode(encoding) # this will raise a unicode error (they're annoying to raise ourselves)
        assert False, 'should have raised an unicode decode error'
def _unquote_c_string(s):
     """replace C-style escape sequences (\n, \", etc.) with real chars."""

     # doing a s.encode('utf-8').decode('unicode_escape') can return an
     # incorrect output with unicode string (both in py2 and py3) the safest way
     # is to match the escape sequences and decoding them alone.
     def decode_match(match):
          return utf8_bytes_string(
               codecs.decode(match.group(0), 'unicode-escape')
          )

     if sys.version_info[0] >= 3 and isinstance(s, bytes):
          return ESCAPE_SEQUENCE_BYTES_RE.sub(decode_match, s)
     else:
          return ESCAPE_SEQUENCE_RE.sub(decode_match, s)
def decode_nfo( buffer ):
    """Decodes a byte string in NFO format (beloved by PC scener groups) from DOS Code Page 437 
    to Unicode."""
    assert utils.is_bytes( buffer )
    return '\n'.join( [''.join( [CP437[y] for y in x] ) for x in buffer.split( b'\r\n' )] )
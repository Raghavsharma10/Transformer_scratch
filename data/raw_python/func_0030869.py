def check_hex_chain(chain):
    """Verify a merkle chain, with hashes hex encoded, to see if the Merkle root can be reproduced.
    """
    return codecs.encode(check_chain([(codecs.decode(i[0], 'hex_codec'), i[1]) for i in chain]), 'hex_codec')
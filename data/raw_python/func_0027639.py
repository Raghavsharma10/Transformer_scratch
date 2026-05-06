def generate_nonce_timestamp():
    """ Generate unique nonce with counter, uuid and rng."""
    global count
    rng = botan.rng().get(30)
    uuid4 = uuid.uuid4().bytes  # 16 byte
    tmpnonce = (bytes(str(count).encode('utf-8'))) + uuid4 + rng
    nonce = tmpnonce[:41]  # 41 byte (328 bit)
    count += 1
    return nonce
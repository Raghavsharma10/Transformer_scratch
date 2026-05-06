def generate_hash(filepath):
    """Public function that reads a local file and generates a SHA256 hash digest for it"""
    fr = FileReader(filepath)
    data = fr.read_bin()
    return _calculate_sha256(data)
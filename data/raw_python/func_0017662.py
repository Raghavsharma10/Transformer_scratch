def download_encrypted_file(work_dir, url, key_path, name):
    """
    Downloads encrypted file from S3

    Input1: Working directory
    Input2: S3 URL to be downloaded
    Input3: Path to key necessary for decryption
    Input4: name of file to be downloaded
    """
    file_path = os.path.join(work_dir, name)
    key = generate_unique_key(key_path, url)
    encoded_key = base64.b64encode(key)
    encoded_key_md5 = base64.b64encode(hashlib.md5(key).digest())
    h1 = 'x-amz-server-side-encryption-customer-algorithm:AES256'
    h2 = 'x-amz-server-side-encryption-customer-key:{}'.format(encoded_key)
    h3 = 'x-amz-server-side-encryption-customer-key-md5:{}'.format(encoded_key_md5)
    try:
        subprocess.check_call(['curl', '-fs', '--retry', '5', '-H', h1, '-H', h2, '-H', h3, url, '-o', file_path])
    except OSError:
        raise RuntimeError('Failed to find "curl". Install via "apt-get install curl"')
    assert os.path.exists(file_path)
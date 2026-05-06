def generate(password, uri):
    """
    generate the pwdhash password for master password and uri or
    domain name.
    """
    realm = extract_domain(uri)
    if password.startswith(_password_prefix):
        password = password[len(_password_prefix):]

    password_hash = b64_hmac_md5(password.encode("utf-8"), realm.encode("utf-8"))
    size = len(password) + len(_password_prefix)
    nonalphanumeric = len(re.findall(r'\W', password)) != 0

    return apply_constraints(password_hash, size, nonalphanumeric)
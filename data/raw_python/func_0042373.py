def main_encrypt(A):
    "Encrypt to recipient list using primary key OR prompted key. Recipients may be IDs or petnames."
    profile = get_profile(A)
    localKeys = profile.get('local keys', [])
    if not localKeys:
        localKeys = [make_lock_securely(warn_only = A.ignore_entropy)]
    else:
        localKeys = [crypto.UserLock.private_from_b64(k['private_key']) for k in localKeys]
    # First key is considered "main"
    userKey = localKeys[0]
    print("User ID:", userKey.userID)
    if not os.path.exists(A.path):
        error_out("File or directory '{}' does not exist.".format(A.path))
    # Create, fetch or error out for recipient list:
    recipients = resolve_recipients(profile, A.recipient)
    recipients.append(userKey)
    print("Recipients:", *set(k.userID if isinstance(k, crypto.UserLock) else k for k in recipients))
    # Do files OR folders
    if os.path.isfile(A.path):
        crypted = encrypt_file(A.path, userKey, recipients)
    elif os.path.isdir(A.path):
        crypted = encrypt_folder(A.path, userKey, recipients)
    else:
        error_out("Specified path '{}' is neither a file nor a folder.".format(A.path))
    if A.base64:
        crypted = crypto.b64encode(crypted)
    if not A.output:
        A.output = hex(int.from_bytes(os.urandom(6),'big'))[2:] + ".minilock"
    print("Saving output to", A.output)
    with open(A.output, "wb") as O:
        O.write(crypted)
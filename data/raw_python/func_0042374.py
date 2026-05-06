def main_decrypt(A):
    "Get all local keys OR prompt user for key, then attempt to decrypt with each."
    profile = get_profile(A)
    localKeys = profile.get('local keys', [])
    if not localKeys:
        localKeys = [make_lock_securely(warn_only = A.ignore_entropy)]
    else:
        localKeys = [crypto.UserLock.private_from_b64(k['private_key']) for k in localKeys]
    if not os.path.exists(A.path):
        error_out("File or directory '{}' does not exist.".format(A.path))
    if os.path.isfile(A.path):
        for k in localKeys:
            print("Attempting decryption with:", k.userID)
            try:
                filename, senderID, decrypted = decrypt_file(A.path, k, base64 = A.base64)
                break
            except ValueError as E:
                pass
        else:
            error_out("Failed to decrypt with all available keys.")
    else:
        error_out("Specified path '{}' is not a file.".format(A.path))
    print("Decrypted file from", senderID)
    print("Saving output to", filename)
    with open(filename, "wb") as O:
        O.write(decrypted)
def make_lock_securely(email = None, warn_only = False):
    "Terminal oriented; produces a prompt for user input of email and password. Returns crypto.UserLock."
    email = email or input("Please provide email address: ")
    while True:
        passphrase = getpass.getpass("Please type a secure passphrase (with spaces): ")
        ok, score = check_passphrase(passphrase, email)
        if ok: break
        print("Insufficiently strong passphrase; has {entropy} bits of entropy, could be broken in {crack_time_display}".format(**score))
        if warn_only: break
        print("Suggestion:", make_random_phrase(email))
    key = crypto.UserLock.from_passphrase(email, passphrase)
    return key
def obfuscate(p, action):
    """Obfuscate the auth details to avoid easy snatching.

    It's best to use a throw away account for these alerts to avoid having
    your authentication put at risk by storing it locally.
    """
    key = "ru7sll3uQrGtDPcIW3okutpFLo6YYtd5bWSpbZJIopYQ0Du0a1WlhvJOaZEH"
    s = list()
    if action == 'store':
        if PY2:
            for i in range(len(p)):
                kc = key[i % len(key)]
                ec = chr((ord(p[i]) + ord(kc)) % 256)
                s.append(ec)
            return base64.urlsafe_b64encode("".join(s))
        else:
            return base64.urlsafe_b64encode(p.encode()).decode()
    else:
        if PY2:
            e = base64.urlsafe_b64decode(p)
            for i in range(len(e)):
                kc = key[i % len(key)]
                dc = chr((256 + ord(e[i]) - ord(kc)) % 256)
                s.append(dc)
            return "".join(s)
        else:
            e = base64.urlsafe_b64decode(p)
            return e.decode()
def verify_file(fp, password):
        'Returns whether a scrypt encrypted file is valid.'

        sf = ScryptFile(fp = fp, password = password)
        for line in sf: pass
        sf.close()
        return sf.valid
def generate(self,
                 secret,
                 type='totp',
                 account='alex',
                 issuer=None,
                 algo='sha1',
                 digits=6,
                 init_counter=None):
        """
        https://github.com/google/google-authenticator/wiki/Key-Uri-Format
        """

        args = {}
        uri = 'otpauth://{0}/{1}?{2}'

        try:
            # converts the secret to a 16 cars string
            a = binascii.unhexlify(secret)
            args[SECRET] = base64.b32encode(a).decode('ascii')
        except binascii.Error as ex:
            raise ValueError(str(ex))
        except Exception as ex:
            print(ex)
            raise ValueError('invalid secret format')

        if type not in [TOTP, HOTP]:
            raise ValueError('type should be totp or hotp, got ',
                             type)
        if type != TOTP:
            args['type'] = type

        if algo not in ['sha1', 'sha256', 'sha512']:
            raise ValueError('algo should be sha1, sha256 or sha512, got ',
                             algo)
        if algo != 'sha1':
            args['algorithm'] = algo

        if init_counter is not None:
            if type != HOTP:
                raise ValueError('type should be hotp when ',
                                 'setting init_counter')

            if int(init_counter) < 0:
                raise ValueError('init_counter should be positive')
            args[COUNTER] = int(init_counter)

        digits = int(digits)
        if digits != 6 and digits != 8:
            raise ValueError('digits should be 6 or 8')
        if digits != 6:
            args[DIGITS] = digits

        args[PERIOD] = 30

        account = quote(account)
        if issuer is not None:
            account = quote(issuer) + ':' + account
            args[ISSUER] = issuer

        uri = uri.format(type, account, urlencode(args).replace("+", "%20"))

        return uri
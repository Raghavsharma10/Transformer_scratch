def install_cert(selfsigned):
    """Install a local SSL certificate"""

    _check_root()

    if selfsigned:
        log('Generating self signed (insecure) certificate/key '
            'combination')

        try:
            os.mkdir('/etc/ssl/certs/hfos')
        except FileExistsError:
            pass
        except PermissionError:
            log("Need root (e.g. via sudo) to generate ssl certificate")
            sys.exit(1)

        def create_self_signed_cert():
            """Create a simple self signed SSL certificate"""

            # create a key pair
            k = crypto.PKey()
            k.generate_key(crypto.TYPE_RSA, 1024)

            if os.path.exists(cert_file):
                try:
                    certificate = open(cert_file, "rb").read()
                    old_cert = crypto.load_certificate(crypto.FILETYPE_PEM,
                                                       certificate)
                    serial = old_cert.get_serial_number() + 1
                except (crypto.Error, OSError) as e:
                    log('Could not read old certificate to increment '
                        'serial:', type(e), e, exc=True, lvl=warn)
                    serial = 1
            else:
                serial = 1

            # create a self-signed certificate
            certificate = crypto.X509()
            certificate.get_subject().C = "DE"
            certificate.get_subject().ST = "Berlin"
            certificate.get_subject().L = "Berlin"
            # noinspection PyPep8
            certificate.get_subject().O = "Hackerfleet"
            certificate.get_subject().OU = "Hackerfleet"
            certificate.get_subject().CN = gethostname()
            certificate.set_serial_number(serial)
            certificate.gmtime_adj_notBefore(0)
            certificate.gmtime_adj_notAfter(10 * 365 * 24 * 60 * 60)
            certificate.set_issuer(certificate.get_subject())
            certificate.set_pubkey(k)
            certificate.sign(k, b'sha512')

            open(key_file, "wt").write(str(
                crypto.dump_privatekey(crypto.FILETYPE_PEM, k),
                encoding="ASCII"))

            open(cert_file, "wt").write(str(
                crypto.dump_certificate(crypto.FILETYPE_PEM, certificate),
                encoding="ASCII"))

            open(combined_file, "wt").write(str(
                crypto.dump_certificate(crypto.FILETYPE_PEM, certificate),
                encoding="ASCII") + str(
                crypto.dump_privatekey(crypto.FILETYPE_PEM, k),
                encoding="ASCII"))

        create_self_signed_cert()

        log('Done: Install Cert')
    else:

        # TODO

        log('Not implemented yet. You can build your own certificate and '
            'store it in /etc/ssl/certs/hfos/server-cert.pem - it should '
            'be a certificate with key, as this is used server side and '
            'there is no way to enter a separate key.', lvl=error)
def split_certificate(certificate_path, destination_folder, password=None):
    """Splits a PKCS12 certificate into Base64-encoded DER certificate and key.

    This method splits a potentially password-protected
    `PKCS12 <https://en.wikipedia.org/wiki/PKCS_12>`_ certificate
    (format ``.p12`` or ``.pfx``) into one certificate and one key part, both in
    `pem <https://en.wikipedia.org/wiki/X.509#Certificate_filename_extensions>`_
    format.

    :returns: Tuple of certificate and key string data.
    :rtype: tuple

    """
    try:
        # Attempt Linux and Darwin call first.
        p = subprocess.Popen(
            ["openssl", "version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        sout, serr = p.communicate()
        openssl_executable_version = sout.decode().lower()
        if not (
            openssl_executable_version.startswith("openssl")
            or openssl_executable_version.startswith("libressl")
        ):
            raise BankIDError(
                "OpenSSL executable could not be found. "
                "Splitting cannot be performed."
            )
        openssl_executable = "openssl"
    except Exception:
        # Attempt to call on standard Git for Windows path.
        p = subprocess.Popen(
            ["C:\\Program Files\\Git\\mingw64\\bin\\openssl.exe", "version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        sout, serr = p.communicate()
        if not sout.decode().lower().startswith("openssl"):
            raise BankIDError(
                "OpenSSL executable could not be found. "
                "Splitting cannot be performed."
            )
        openssl_executable = "C:\\Program Files\\Git\\mingw64\\bin\\openssl.exe"

    if not os.path.exists(os.path.abspath(os.path.expanduser(destination_folder))):
        os.makedirs(os.path.abspath(os.path.expanduser(destination_folder)))

    # Paths to output files.
    out_cert_path = os.path.join(
        os.path.abspath(os.path.expanduser(destination_folder)), "certificate.pem"
    )
    out_key_path = os.path.join(
        os.path.abspath(os.path.expanduser(destination_folder)), "key.pem"
    )

    # Use openssl for converting to pem format.
    pipeline_1 = [
        openssl_executable,
        "pkcs12",
        "-in",
        "{0}".format(certificate_path),
        "-passin" if password is not None else "",
        "pass:{0}".format(password) if password is not None else "",
        "-out",
        "{0}".format(out_cert_path),
        "-clcerts",
        "-nokeys",
    ]
    p = subprocess.Popen(
        list(filter(None, pipeline_1)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    p.communicate()
    pipeline_2 = [
        openssl_executable,
        "pkcs12",
        "-in",
        "{0}".format(certificate_path),
        "-passin" if password is not None else "",
        "pass:{0}".format(password) if password is not None else "",
        "-out",
        "{0}".format(out_key_path),
        "-nocerts",
        "-nodes",
    ]
    p = subprocess.Popen(
        list(filter(None, pipeline_2)), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    p.communicate()

    # Return path tuples.
    return out_cert_path, out_key_path
def create(message: str, pubkey: Optional[str] = None, signing_keys: Optional[List[SigningKey]] = None,
               message_comment: Optional[str] = None, signatures_comment: Optional[str] = None) -> str:
        """
        Encrypt a message in ascii armor format, optionally signing it

        :param message: Utf-8 message
        :param pubkey: Public key of recipient for encryption
        :param signing_keys: Optional list of SigningKey instances
        :param message_comment: Optional message comment field
        :param signatures_comment: Optional signatures comment field
        :return:
        """
        # if no public key and no signing key...
        if not pubkey and not signing_keys:
            # We can not create an Ascii Armor Message
            raise MISSING_PUBLIC_KEY_AND_SIGNING_KEY_EXCEPTION

        # keep only one newline at the end of the message
        message = message.rstrip("\n\r") + "\n"

        # create block with headers
        ascii_armor_block = """{begin_message_header}
""".format(begin_message_header=BEGIN_MESSAGE_HEADER)

        # if encrypted message...
        if pubkey:
            # add encrypted message fields
            ascii_armor_block += """{version_field}
""".format(version_field=AsciiArmor._get_version_field())

        # add message comment if specified
        if message_comment:
            ascii_armor_block += """{comment_field}
""".format(comment_field=AsciiArmor._get_comment_field(message_comment))

        # blank line separator
        ascii_armor_block += '\n'

        if pubkey:
            # add encrypted message
            pubkey_instance = PublicKey(pubkey)
            base64_encrypted_message = base64.b64encode(pubkey_instance.encrypt_seal(message))  # type: bytes
            ascii_armor_block += """{base64_encrypted_message}
""".format(base64_encrypted_message=base64_encrypted_message.decode('utf-8'))
        else:
            # remove trailing spaces
            message = AsciiArmor._remove_trailing_spaces(message)

            # add dash escaped message to ascii armor content
            ascii_armor_block += AsciiArmor._dash_escape_text(message)

        # if no signature...
        if signing_keys is None:
            # add message tail
            ascii_armor_block += END_MESSAGE_HEADER
        else:
            # add signature blocks and close block on last signature
            count = 1
            for signing_key in signing_keys:
                ascii_armor_block += AsciiArmor._get_signature_block(message, signing_key, count == len(signing_keys),
                                                                     signatures_comment)
                count += 1

        return ascii_armor_block
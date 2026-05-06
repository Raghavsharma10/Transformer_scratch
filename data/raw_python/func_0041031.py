def parse(ascii_armor_message: str, signing_key: Optional[SigningKey] = None,
              sender_pubkeys: Optional[List[str]] = None) -> dict:
        """
        Return a dict with parsed content (decrypted message, signature validation) ::

            {
               'message':
                   {
                       'fields': {},
                       'content': str,
                    },
               'signatures': [
                   {'pubkey': str, 'valid': bool, fields: {}}
               ]
           }

        :param ascii_armor_message: The Ascii Armor Message Block including BEGIN and END headers
        :param signing_key: Optional Libnacl SigningKey instance to decrypt message
        :param sender_pubkeys: Optional sender's public keys list to verify signatures
        :exception libnacl.CryptError: Raise an exception if keypair fail to decrypt the message
        :exception MissingSigningKeyException: Raise an exception if no keypair given for encrypted message

        :return:
        """
        # regex patterns
        regex_begin_message = compile(BEGIN_MESSAGE_HEADER)
        regex_end_message = compile(END_MESSAGE_HEADER)
        regex_begin_signature = compile(BEGIN_SIGNATURE_HEADER)
        regex_end_signature = compile(END_SIGNATURE_HEADER)
        regex_fields = compile("^(Version|Comment): (.+)$")

        # trim message to get rid of empty lines
        ascii_armor_message = ascii_armor_message.strip(" \t\n\r")

        # init vars
        parsed_result = {
            'message':
                {
                    'fields': {},
                    'content': '',
                },
            'signatures': []
        }  # type: Dict[str, Any]
        cursor_status = 0
        message = ''
        signatures_index = 0

        # parse each line...
        for line in ascii_armor_message.splitlines(True):

            # if begin message header detected...
            if regex_begin_message.match(line):
                cursor_status = ON_MESSAGE_FIELDS
                continue

            # if we are on the fields lines...
            if cursor_status == ON_MESSAGE_FIELDS:
                # parse field
                m = regex_fields.match(line.strip())
                if m:
                    # capture field
                    msg_field_name = m.groups()[0]
                    msg_field_value = m.groups()[1]
                    parsed_result['message']['fields'][msg_field_name] = msg_field_value
                    continue

                # if blank line...
                if line.strip("\n\t\r ") == '':
                    cursor_status = ON_MESSAGE_CONTENT
                    continue

            # if we are on the message content lines...
            if cursor_status == ON_MESSAGE_CONTENT:

                # if a header is detected, end of message content...
                if line.startswith(HEADER_PREFIX):

                    # if field Version is present, the message is encrypted...
                    if 'Version' in parsed_result['message']['fields']:

                        # If keypair instance to decrypt not given...
                        if signing_key is None:
                            # SigningKey keypair is mandatory to decrypt the message...
                            raise PARSER_MISSING_SIGNING_KEY_EXCEPTION

                        # decrypt message with secret key from keypair
                        message = AsciiArmor._decrypt(message, signing_key)

                    # save message content in result
                    parsed_result['message']['content'] = message

                    # if message end header...
                    if regex_end_message.match(line):
                        # stop parsing
                        break

                    # if signature begin header...
                    if regex_begin_signature.match(line):
                        # add signature dict in list
                        parsed_result['signatures'].append({
                            'fields': {}
                        })
                        cursor_status = ON_SIGNATURE_FIELDS
                        continue
                else:
                    # if field Version is present, the message is encrypted...
                    if 'Version' in parsed_result['message']['fields']:
                        # concatenate encrypted line to message content
                        message += line
                    else:
                        # concatenate cleartext striped dash escaped line to message content
                        message += AsciiArmor._parse_dash_escaped_line(line)

            # if we are on a signature fields zone...
            if cursor_status == ON_SIGNATURE_FIELDS:

                # parse field
                m = regex_fields.match(line.strip())
                if m:
                    # capture field
                    sig_field_name = m.groups()[0]
                    sig_field_value = m.groups()[1]
                    parsed_result['signatures'][signatures_index]['fields'][sig_field_name] = sig_field_value
                    continue

                # if blank line...
                if line.strip("\n\t\r ") == '':
                    cursor_status = ON_SIGNATURE_CONTENT
                    continue

            # if we are on the signature content...
            if cursor_status == ON_SIGNATURE_CONTENT:

                # if no public keys provided...
                if sender_pubkeys is None:
                    # raise exception
                    raise PARSER_MISSING_PUBLIC_KEYS_EXCEPTION

                # if end signature header detected...
                if regex_end_signature.match(line):
                    # end of parsing
                    break

                # if begin signature header detected...
                if regex_begin_signature.match(line):
                    signatures_index += 1
                    cursor_status = ON_SIGNATURE_FIELDS
                    continue

                for pubkey in sender_pubkeys:
                    verifier = VerifyingKey(pubkey)
                    signature = base64.b64decode(line)
                    parsed_result['signatures'][signatures_index]['pubkey'] = pubkey
                    try:
                        libnacl.crypto_sign_verify_detached(signature, message, verifier.vk)
                        parsed_result['signatures'][signatures_index]['valid'] = True
                    except ValueError:
                        parsed_result['signatures'][signatures_index]['valid'] = False

        return parsed_result
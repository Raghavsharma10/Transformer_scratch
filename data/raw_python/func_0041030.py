def _get_signature_block(message: str, signing_key: SigningKey, close_block: bool = True,
                             comment: Optional[str] = None) -> str:
        """
        Return a signature block

        :param message: Message (not encrypted!) to sign
        :param signing_key: The libnacl SigningKey instance of the keypair
        :param close_block: Optional flag to close the signature block with the signature tail header
        :param comment: Optional comment field content
        :return:
        """
        base64_signature = base64.b64encode(signing_key.signature(message))

        block = """{begin_signature_header}
{version_field}
""".format(begin_signature_header=BEGIN_SIGNATURE_HEADER, version_field=AsciiArmor._get_version_field())

        # add message comment if specified
        if comment:
            block += """{comment_field}
""".format(comment_field=AsciiArmor._get_comment_field(comment))

        # blank line separator
        block += '\n'

        block += """{base64_signature}
""".format(base64_signature=base64_signature.decode('utf-8'))

        if close_block:
            block += END_SIGNATURE_HEADER

        return block
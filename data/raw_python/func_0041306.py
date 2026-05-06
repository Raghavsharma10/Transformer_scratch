def get_certification_document(current_block: dict, self_cert_document: Identity, from_pubkey: str) -> Certification:
    """
    Create and return a Certification document

    :param current_block: Current block data
    :param self_cert_document: Identity document
    :param from_pubkey: Pubkey of the certifier

    :rtype: Certification
    """
    # construct Certification Document
    return Certification(version=10, currency=current_block['currency'], pubkey_from=from_pubkey,
                                  identity=self_cert_document,
                                  timestamp=BlockUID(current_block['number'], current_block['hash']), signature="")
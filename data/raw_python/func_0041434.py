def get_transaction_document(current_block: dict, source: dict, from_pubkey: str, to_pubkey: str) -> Transaction:
    """
    Return a Transaction document

    :param current_block: Current block infos
    :param source: Source to send
    :param from_pubkey: Public key of the issuer
    :param to_pubkey: Public key of the receiver

    :return: Transaction
    """
    # list of inputs (sources)
    inputs = [
        InputSource(
            amount=source['amount'],
            base=source['base'],
            source=source['type'],
            origin_id=source['identifier'],
            index=source['noffset']
        )
    ]

    # list of issuers of the inputs
    issuers = [
        from_pubkey
    ]

    # list of unlocks of the inputs
    unlocks = [
        Unlock(
            # inputs[index]
            index=0,
            # unlock inputs[index] if signatures[0] is from public key of issuers[0]
            parameters=[SIGParameter(0)]
        )
    ]

    # lists of outputs
    outputs = [
        OutputSource(amount=source['amount'], base=source['base'], condition="SIG({0})".format(to_pubkey))
    ]

    transaction = Transaction(
        version=TRANSACTION_VERSION,
        currency=current_block['currency'],
        blockstamp=BlockUID(current_block['number'], current_block['hash']),
        locktime=0,
        issuers=issuers,
        inputs=inputs,
        unlocks=unlocks,
        outputs=outputs,
        comment='',
        signatures=[]
    )

    return transaction
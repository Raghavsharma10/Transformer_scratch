def from_signed_raw(cls: Type[TransactionType], raw: str) -> TransactionType:
        """
        Return a Transaction instance from a raw string format

        :param raw: Raw string format

        :return:
        """
        lines = raw.splitlines(True)
        n = 0

        version = int(Transaction.parse_field("Version", lines[n]))
        n += 1

        Transaction.parse_field("Type", lines[n])
        n += 1

        currency = Transaction.parse_field("Currency", lines[n])
        n += 1

        blockstamp = None  # type: Optional[BlockUID]
        if version >= 3:
            blockstamp = BlockUID.from_str(Transaction.parse_field("Blockstamp", lines[n]))
            n += 1

        locktime = Transaction.parse_field("Locktime", lines[n])
        n += 1

        issuers = []
        inputs = []
        unlocks = []
        outputs = []
        signatures = []

        if Transaction.re_issuers.match(lines[n]):
            n += 1
            while Transaction.re_inputs.match(lines[n]) is None:
                issuer = Transaction.parse_field("Pubkey", lines[n])
                issuers.append(issuer)
                n += 1

        if Transaction.re_inputs.match(lines[n]):
            n += 1
            while Transaction.re_unlocks.match(lines[n]) is None:
                input_source = InputSource.from_inline(version, lines[n])
                inputs.append(input_source)
                n += 1

        if Transaction.re_unlocks.match(lines[n]):
            n += 1
            while Transaction.re_outputs.match(lines[n]) is None:
                unlock = Unlock.from_inline(lines[n])
                unlocks.append(unlock)
                n += 1

        if Transaction.re_outputs.match(lines[n]) is not None:
            n += 1
            while not Transaction.re_comment.match(lines[n]):
                _output = OutputSource.from_inline(lines[n])
                outputs.append(_output)
                n += 1

        comment = Transaction.parse_field("Comment", lines[n])
        n += 1

        if Transaction.re_signature.match(lines[n]) is not None:
            while n < len(lines):
                sign = Transaction.parse_field("Signature", lines[n])
                signatures.append(sign)
                n += 1

        return cls(version, currency, blockstamp, locktime, issuers, inputs, unlocks, outputs,
                   comment, signatures)
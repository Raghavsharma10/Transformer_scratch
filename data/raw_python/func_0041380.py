def from_compact(cls: Type[TransactionType], currency: str, compact: str) -> TransactionType:
        """
        Return Transaction instance from compact string format

        :param currency: Name of the currency
        :param compact: Compact format string
        :return:
        """
        lines = compact.splitlines(True)
        n = 0

        header_data = Transaction.re_header.match(lines[n])
        if header_data is None:
            raise MalformedDocumentError("Compact TX header")
        version = int(header_data.group(1))
        issuers_num = int(header_data.group(2))
        inputs_num = int(header_data.group(3))
        unlocks_num = int(header_data.group(4))
        outputs_num = int(header_data.group(5))
        has_comment = int(header_data.group(6))
        locktime = int(header_data.group(7))
        n += 1

        blockstamp = None  # type: Optional[BlockUID]
        if version >= 3:
            blockstamp = BlockUID.from_str(Transaction.parse_field("CompactBlockstamp", lines[n]))
            n += 1

        issuers = []
        inputs = []
        unlocks = []
        outputs = []
        signatures = []
        for i in range(0, issuers_num):
            issuer = Transaction.parse_field("Pubkey", lines[n])
            issuers.append(issuer)
            n += 1

        for i in range(0, inputs_num):
            input_source = InputSource.from_inline(version, lines[n])
            inputs.append(input_source)
            n += 1

        for i in range(0, unlocks_num):
            unlock = Unlock.from_inline(lines[n])
            unlocks.append(unlock)
            n += 1

        for i in range(0, outputs_num):
            output_source = OutputSource.from_inline(lines[n])
            outputs.append(output_source)
            n += 1

        comment = ""
        if has_comment == 1:
            data = Transaction.re_compact_comment.match(lines[n])
            if data:
                comment = data.group(1)
                n += 1
            else:
                raise MalformedDocumentError("Compact TX Comment")

        while n < len(lines):
            data = Transaction.re_signature.match(lines[n])
            if data:
                signatures.append(data.group(1))
                n += 1
            else:
                raise MalformedDocumentError("Compact TX Signatures")

        return cls(version, currency, blockstamp, locktime, issuers, inputs, unlocks, outputs, comment, signatures)
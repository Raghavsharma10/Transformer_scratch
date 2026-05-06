def from_bma_history(cls: Type[TransactionType], currency: str, tx_data: Dict) -> TransactionType:
        """
        Get the transaction instance from json

        :param currency: the currency of the tx
        :param tx_data: json data of the transaction

        :return:
        """
        tx_data = tx_data.copy()
        tx_data["currency"] = currency
        for data_list in ('issuers', 'outputs', 'inputs', 'unlocks', 'signatures'):
            tx_data['multiline_{0}'.format(data_list)] = '\n'.join(tx_data[data_list])
        if tx_data["version"] >= 3:
            signed_raw = """Version: {version}
Type: Transaction
Currency: {currency}
Blockstamp: {blockstamp}
Locktime: {locktime}
Issuers:
{multiline_issuers}
Inputs:
{multiline_inputs}
Unlocks:
{multiline_unlocks}
Outputs:
{multiline_outputs}
Comment: {comment}
{multiline_signatures}
""".format(**tx_data)
        else:
            signed_raw = """Version: {version}
Type: Transaction
Currency: {currency}
Locktime: {locktime}
Issuers:
{multiline_issuers}
Inputs:
{multiline_inputs}
Unlocks:
{multiline_unlocks}
Outputs:
{multiline_outputs}
Comment: {comment}
{multiline_signatures}
""".format(**tx_data)
        return cls.from_signed_raw(signed_raw)
def parse_transaction_id(self, data):
        "return transaction_id"
        if data[0] == TDS_ERROR_TOKEN:
            raise self.parse_error('begin()', data)
        t, data = _parse_byte(data)
        assert t == TDS_ENVCHANGE_TOKEN
        _, data = _parse_int(data, 2)   # packet length
        e, data = _parse_byte(data)
        assert e == TDS_ENV_BEGINTRANS
        ln, data = _parse_byte(data)
        assert ln == 8                  # transaction id length
        return data[:ln], data[ln:]
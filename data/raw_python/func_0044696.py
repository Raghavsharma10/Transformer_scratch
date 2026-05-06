async def withdraw(self, *args, **kwargs):
        """
        Withdraw funds to user wallet

        Accepts:
            - coinid [string] (blockchain id (example: BTCTEST, LTCTEST))
            - address [string] withdrawal address (in hex for tokens)
            - amount [int]     withdrawal amount multiplied by decimals_k (10**8)
        Returns dictionary with following fields:
            - txid [string]
        """
        try:
            super().reload_connections()
        except Exception as e:
            return WithdrawValidator.error_500(str(e))

        coinid = kwargs.get("coinid")
        address = kwargs.get("address")
        amount = int(kwargs.get("amount"))
        txid = None
        connection = self.connections[coinid]

        if coinid in ['BTCTEST', 'LTCTEST', 'QTUMTEST', 'BTC', 'LTC', 'QTUM']:
            try:
                txid = connection.sendtoaddress(address, str(amount / decimals_k))
            except Exception as e:
                return WithdrawValidator.error_400(str(e))
        elif coinid in ['ETH', 'ETHRINKEBY', 'ETHROPSTEN']:
            address = Web3.toChecksumAddress(address)
            try:
                txid = connection.eth.sendTransaction(
                    {'to': address, 'from': hot_wallets[coinid], 'value': Web3.toWei(amount / decimals_k, 'ether')}
                )
                txid = encode_hex(txid)[0].decode()
            except Exception as e:
                return WithdrawValidator.error_500(str(e))
        else:
            token = await self.db.available_tokens.find_one({'_id': coinid})
            if token is None:
                return WithdrawValidator.error_500('Unsupported coinid')

            elif token['blockchain'] in ('QTUM', 'QTUMTEST'):
                connection = self.connections[coinid]
                address = Bip32Addresses.address_to_hex(address)
                handler = Qrc20.from_connection(
                    connection,
                    token['contract_address'],
                    erc20_abi
                )
                handler.set_send_params({
                    'gasLimit': transfer_settings[token['blockchain']]['gasLimit'],
                    'gasPrice': transfer_settings[token['blockchain']]['gasPrice'],
                    'sender': hot_wallets[coinid]
                })
                print(hot_wallets[coinid])
                try:
                    txid = handler.transfer(address, amount)['txid']
                except Exception as e:
                    return WithdrawValidator.error_500(str(e))

            elif token['blockchain'] in ('ETHRINKEBY', 'ETH'):
                connection = self.connections[coinid]
                address = Web3.toChecksumAddress(address)
                handler = Erc20.from_connection(
                    connection,
                    token['contract_address'],
                    erc20_abi
                )
                handler.set_send_params({
                    'gasLimit': transfer_settings[token['blockchain']]['gasLimit'],
                    'gasPrice': transfer_settings[token['blockchain']]['gasPrice'],
                    'sender': hot_wallets[coinid]})
                try:
                    txid = handler.transfer(address, amount)['txid']
                except Exception as e:
                    return WithdrawValidator.error_500(str(e))

        await self.db.executed_withdraws.insert_one({
            'coinid': coinid,
            'address': address,
            'amount': amount,
            'txid': txid,
            'timestamp': datetime.datetime.utcnow(),
            'execution_time': datetime.datetime.utcnow()
        })
        return {'txid': txid}
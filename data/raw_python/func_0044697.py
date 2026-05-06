async def withdraw_custom_token(self, *args, **kwargs):
        """
        Withdraw custom token to user wallet

        Accepts:
            - address [hex string] (withdrawal address in hex form)
            - amount [int] withdrawal amount multiplied by decimals_k (10**8)
            - blockchain [string]  token's blockchain (QTUMTEST, ETH)
            - contract_address [hex string] token contract address
        Returns dictionary with following fields:
            - txid [string]
        """
        try:
            super().reload_connections()
        except Exception as e:
            return WithdrawValidator.error_500(str(e))

        address = kwargs.get("address")
        amount = kwargs.get("amount")
        blockchain = kwargs.get("blockchain")
        contract_address = kwargs.get("contract_address")

        await self.db.withdraw_custom_token_requests.insert_one({
            'address': address,
            'amount': amount,
            'blockchain': blockchain,
            'contract_address': contract_address,
            'timestamp': datetime.datetime.utcnow()
        })

        connection = self.connections[blockchain]

        if blockchain in ['QTUMTEST', 'QTUM']:
            address = Bip32Addresses.address_to_hex(address)
            handler = Qrc20.from_connection(connection, contract_address, erc20_abi)
            handler.set_send_params({
                'gasLimit': transfer_settings[blockchain]['gasLimit'],
                'gasPrice': transfer_settings[blockchain]['gasPrice'],
                'sender': hot_wallets[blockchain]
            })
            try:
                txid = handler.transfer(address, amount)['txid']
            except Exception as e:
                return WithdrawValidator.error_500(str(e))

        elif blockchain in ['ETH', 'ETHRINKEBY', 'ETHROPSTEN']:
            address = Web3.toChecksumAddress(address)
            contract_address = Web3.toChecksumAddress(contract_address)
            handler = Erc20.from_connection(connection, contract_address, erc20_abi)
            handler.set_send_params({
                'gasLimit': transfer_settings[blockchain]['gasLimit'],
                'gasPrice': transfer_settings[blockchain]['gasPrice'],
                'sender': hot_wallets[blockchain]
            })
            try:
                txid = handler.transfer(address, amount)['txid']
            except Exception as e:
                return WithdrawValidator.error_500(str(e))
        else:
            return WithdrawValidator.error_403('Unsupported blockchain')

        return {'txid': txid}
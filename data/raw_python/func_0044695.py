async def withdraw_bulk(self, *args, **kwargs):
        """
       Withdraw funds requests to user wallet

       Accepts:
           - coinid [string] (blockchain id (example: BTCTEST, LTCTEST))
           - address [string] withdrawal address (in hex for tokens)
           - amount [int]     withdrawal amount multiplied by decimals_k (10**8)
       Returns dictionary with following fields:
           - success [bool]
       """
        await self.db.withdraw_requests.insert_one({
            'coinid': kwargs.get("coinid"),
            'address': kwargs.get("address"),
            'amount': int(kwargs.get("amount")),
            'timestamp': datetime.datetime.utcnow()
        })

        return {'success': True}
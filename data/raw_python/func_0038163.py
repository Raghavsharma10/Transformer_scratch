def record_tx(self, origin, destination, amount,
                  outcome, destination_id=None):
        """Records a transaction in the database.

        Args:
          origin (str): user_id of the sender
          destination (str): coin address or user_id of the recipient
          amount (str, Decimal, number): amount to send
          outcome (str, bool): the transaction hash if this is a "sendfrom"
                               transaction; for "move", True if successful,
                               False otherwise
          destination_id (str): the destination account label ("move" only)

        Returns:
          str or bool: the outcome (input) argument

        """
        # "move" commands
        if destination_id:
            tx = db.Transaction(
                txtype="move",
                from_user_id=origin,
                to_user_id=destination_id,
                txdate=datetime.now(),
                amount=amount,
                currency=COINS[self.coin]["ticker"],
                to_coin_address=destination,
            )

        # "sendfrom" commands
        else:
            self.logger.debug(self.gettransaction(outcome))
            confirmations = self.gettransaction(outcome)["confirmations"]
            last_confirmation = datetime.now() if confirmations else None
            tx = db.Transaction(
                txtype="sendfrom",
                from_user_id=origin,
                txhash=outcome,
                txdate=datetime.now(),
                amount=amount,
                currency=COINS[self.coin]["ticker"],
                to_coin_address=destination,
                confirmations=confirmations,
                last_confirmation=last_confirmation
            )
        db.session.add(tx)
        db.session.commit()
        return outcome
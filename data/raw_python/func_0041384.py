def is_simple(tx: Transaction) -> bool:
        """
        Filter a transaction and checks if it is a basic one
        A simple transaction is a tx which has only one issuer
        and two outputs maximum. The unlocks must be done with
        simple "SIG" functions, and the outputs must be simple
        SIG conditions.

        :param tx: the transaction to check

        :return: True if a simple transaction
        """
        simple = True
        if len(tx.issuers) != 1:
            simple = False
        for unlock in tx.unlocks:
            if len(unlock.parameters) != 1:
                simple = False
            elif type(unlock.parameters[0]) is not SIGParameter:
                simple = False
        for o in tx.outputs:
            # if right condition is not None...
            if getattr(o.condition, 'right', None):
                simple = False
                # if left is not SIG...
            elif type(o.condition.left) is not output.SIG:
                simple = False

        return simple
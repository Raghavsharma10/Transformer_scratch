def watch_pending_transactions(self, callback):
        '''
        Callback will receive one argument: the transaction object just observed

        This is equivalent to `eth.filter('pending')`
        '''
        self.pending_tx_watchers.append(callback)
        if len(self.pending_tx_watchers) == 1:
            eth.filter('pending').watch(self._new_pending_tx)
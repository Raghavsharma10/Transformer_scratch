def _start_watching_blocks(self):
        '''
        Internal: call immediately after registering a block watcher
        If the new watcher is the first, then start watching web3 remotely
        '''
        if sum(map(len, (self.solid_watchers, self.block_watchers))) == 1:
            eth.filter('latest').watch(self._new_block)
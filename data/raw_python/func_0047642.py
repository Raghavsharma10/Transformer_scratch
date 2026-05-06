def __push_symbol(self, symbol):
        '''Ask the websocket for a symbol push. Gets instrument, orderBook, quote, and trade'''
        self.__send_command("getSymbol", symbol)
        while not {'instrument', 'trade', 'orderBook25'} <= set(self.data):
            sleep(0.1)
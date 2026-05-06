def checkStock(self):
        """check stocks in preference"""
        if not self.preferences:
            logger.debug("no preferences")
            return None
        soup = BeautifulSoup(
            self.xpath(path['stock-table'])[0].html, "html.parser")
        count = 0
        # iterate through product in left panel
        for product in soup.select("div.tradebox"):
            prod_name = product.select("span.instrument-name")[0].text
            stk_name = [x for x in self.preferences
                        if x.lower() in prod_name.lower()]
            if not stk_name:
                continue
            name = prod_name
            if not [x for x in self.stocks if x.product == name]:
                self.stocks.append(Stock(name))
            stock = [x for x in self.stocks if x.product == name][0]
            if 'tradebox-market-closed' in product['class']:
                stock.market = False
            if not stock.market:
                logger.debug("market closed for %s" % stock.product)
                continue
            sell_price = product.select("div.tradebox-price-sell")[0].text
            buy_price = product.select("div.tradebox-price-buy")[0].text
            sent = int(product.select(path['sent'])[0].text.strip('%')) / 100
            stock.new_rec([sell_price, buy_price, sent])
            count += 1
        logger.debug(f"added %d stocks" % count)
        return self.stocks
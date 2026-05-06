def clearPrefs(self):
        """clear the left panel and preferences"""
        self.preferences.clear()
        tradebox_num = len(self.css('div.tradebox'))
        for i in range(tradebox_num):
            self.xpath(path['trade-box'])[0].right_click()
            self.css1('div.item-trade-contextmenu-list-remove').click()
        logger.info("cleared preferences")
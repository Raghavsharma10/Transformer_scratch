def addPrefs(self, prefs=[]):
        """add preference in self.preferences"""
        if len(prefs) == len(self.preferences) == 0:
            logger.debug("no preferences")
            return None
        self.preferences.extend(prefs)
        self.css1(path['search-btn']).click()
        count = 0
        for pref in self.preferences:
            self.css1(path['search-pref']).fill(pref)
            self.css1(path['pref-icon']).click()
            btn = self.css1('div.add-to-watchlist-popup-item .icon-wrapper')
            if not self.css1('svg', btn)['class'] is None:
                btn.click()
                count += 1
            # remove window
            self.css1(path['pref-icon']).click()
        # close finally
        self.css1(path['back-btn']).click()
        self.css1(path['back-btn']).click()
        logger.debug("updated %d preferences" % count)
        return self.preferences
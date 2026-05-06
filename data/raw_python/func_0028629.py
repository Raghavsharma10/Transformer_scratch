def tap_key(self, key, complementKey=None) :
		"""Presses the specified `key`. The `complementKey` defines the key to hold
		when pressing the specified `key`. For example, you could use ${VK_TAB} as `key` and
		use ${VK_SHIFT} as `complementKey' in order to press Shift + Tab (back tab)

		| =Argument= | =Description= | =Example= |
		| key | the key to press | ${VK_F4} |
		| complementKey | the key to hold while pressing the key passed in previous argument | ${VK_ALT} |"""

		driver = self._current_browser()

		if (complementKey is not None) :
			ActionChains(driver).key_down(complementKey).send_keys(key).key_up(complementKey).perform()

		else :
			ActionChains(driver).send_keys(key).perform()
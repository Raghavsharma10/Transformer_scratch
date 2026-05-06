def get_all_rules(cls):
        "Load all available Adblock rules."
        from adblockparser import AdblockRules
        
        raw_rules = []
        for url in [
            config.ADBLOCK_EASYLIST_URL, config.ADBLOCK_EXTRALIST_URL]:
            raw_rules.extend(cls.load_raw_rules(url))

        rules = AdblockRules(raw_rules)
        return rules
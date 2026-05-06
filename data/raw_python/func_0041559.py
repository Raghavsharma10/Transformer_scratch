def apply_all_rules(self, *args, **kwargs):
        """cycle through all rules and apply them all without regard to
        success or failure

        returns:
             True - since success or failure is ignored"""
        for x in self.rules:
            self._quit_check()
            if self.config.chatty_rules:
                self.config.logger.debug(
                    'apply_all_rules: %s',
                    to_str(x.__class__)
                )
            predicate_result, action_result = x.act(*args, **kwargs)
            if self.config.chatty_rules:
                self.config.logger.debug(
                    '               : pred - %s; act - %s',
                    predicate_result,
                    action_result
                )
        return True
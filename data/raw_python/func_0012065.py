def generate(self, signature_data):
        """Takes data and returns a signature

        :arg dict signature_data: data to use to generate a signature

        :returns: ``Result`` instance

        """
        result = Result()

        for rule in self.pipeline:
            rule_name = rule.__class__.__name__

            try:
                if rule.predicate(signature_data, result):
                    rule.action(signature_data, result)

            except Exception as exc:
                if self.error_handler:
                    self.error_handler(
                        signature_data,
                        exc_info=sys.exc_info(),
                        extra={'rule': rule_name}
                    )
                result.info(rule_name, 'Rule failed: %s', exc)

        return result
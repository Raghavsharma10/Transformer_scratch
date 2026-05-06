def expand(self, expression):
        """Expands logical constructions."""
        self.logger.debug("expand : expression %s", str(expression))
        if not is_string(expression):
            return expression

        result = self._pattern.sub(lambda var: str(self._variables[var.group(1)]), expression)

        result = result.strip()
        self.logger.debug('expand : %s - result : %s', expression, result)

        if is_number(result):
            if result.isdigit():
                self.logger.debug('     expand is integer !!!')
                return int(result)
            else:
                self.logger.debug('     expand is float !!!')
                return float(result)
        return result
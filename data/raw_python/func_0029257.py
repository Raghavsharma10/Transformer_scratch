def _check_animation(self, last, last_level, gen):
        """Internal helper function to check the animation."""
        tmp_ = Counter()
        results = Counter()
        name, level, block = last, last_level, last
        others = False
        while name in self.ANIMATION and level >= last_level:
            if name in self.LOOP:
                if block != last:
                    count = self.check_results(tmp_)
                    if count > -1:
                        results[count] += 1
                    tmp_.clear()
                tmp_['last'] += 1

            for attribute in ('costume', 'orientation', 'position', 'size'):
                if (name, 'relative') in self.BLOCKMAPPING[attribute]:
                    tmp_[(attribute, 'relative')] += 1
                elif (name, 'absolute') in self.BLOCKMAPPING[attribute]:
                    tmp_[(attribute, 'absolute')] += 1
            if name in self.TIMING:
                tmp_['timing'] += 1

            last_level = level
            name, level, block = next(gen, ('', 0, ''))
            # allow some exceptions
            if name not in self.ANIMATION and name != '':
                if not others:
                    if block.type.shape != 'stack':
                        last_level = level
                        (name, level, block) = next(gen, ('', 0, ''))
                        others = True
        count = self.check_results(tmp_)
        if count > -1:
            results[count] += 1
        return gen, results
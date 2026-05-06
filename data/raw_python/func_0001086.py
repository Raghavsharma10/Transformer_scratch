def pos3(self):
        ''' Use pos-sc1-sc2 as POS '''
        parts = [self.pos]
        if self.sc1 and self.sc1 != '*':
            parts.append(self.sc1)
            if self.sc2 and self.sc2 != '*':
                parts.append(self.sc2)
        return '-'.join(parts)
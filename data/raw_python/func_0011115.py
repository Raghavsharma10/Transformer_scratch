def repr_part(self):
        """FmtStr repr is build by concatenating these."""
        def pp_att(att):
            if att == 'fg': return FG_NUMBER_TO_COLOR[self.atts[att]]
            elif att == 'bg': return 'on_' + BG_NUMBER_TO_COLOR[self.atts[att]]
            else: return att
        atts_out = dict((k, v) for (k, v) in self.atts.items() if v)
        return (''.join(pp_att(att)+'(' for att in sorted(atts_out))
                + (repr(self.s) if PY3 else repr(self.s)[1:]) + ')'*len(atts_out))
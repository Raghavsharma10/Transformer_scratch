def get_prototype(self, alias):
        """
        RETURN ALL INDEXES THAT ARE INTENDED TO BE GIVEN alias, BUT HAVE NO
        ALIAS YET BECAUSE INCOMPLETE
        """
        output = sort([
            a.index
            for a in self.get_aliases()
            if re.match(re.escape(alias) + "\\d{8}_\\d{6}", a.index) and not a.alias
        ])
        return output
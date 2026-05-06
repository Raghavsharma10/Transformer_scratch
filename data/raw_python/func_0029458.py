def fill(self, postf_un_ops: str):
        """
        Insert:
          * math styles
          * other styles
          * unary prefix operators without brackets
          * defaults
        """
        for op, dic in self.ops.items():
            if 'postf' not in dic:
                dic['postf'] = self.postf
        self.ops = OrderedDict(
            self.styles.spec(postf_un_ops) +
            self.other_styles.spec(postf_un_ops) +
            self.pref_un_greedy.spec() +
            list(self.ops.items())
        )
        for op, dic in self.ops.items():
            dic['postf'] = re.compile(dic['postf'])
        self.regex = _search_regex(self.ops, self.regex_pat)
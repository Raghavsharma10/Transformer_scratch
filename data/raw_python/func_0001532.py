def align(self):
        '''
        Every step we have 3 choices:
        1) Move pointer witness a --> omission
        2) Move pointer witness b --> addition
        3) Move pointer of both witness a/b  --> match
        Note: a replacement is omission followed by an addition or the other way around

        Choice 1 and 2 are only possible if token a and b are not a match OR when tokens are repeated.
        For now I ignore token repetition..
        '''
        # extract tokens from witness (note that this can be done in a streaming manner if desired)
        tokens_a = self.witness_a.tokens()
        tokens_b = self.witness_b.tokens()
        # create virtual decision tree (nodes are created on demand)
        # see above
        # create start node
        start = DecisionTreeNode(self)

        # search the decision tree
        result = self.tree.search(start)
        print(result)

        pass
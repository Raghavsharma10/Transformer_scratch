def _add_to_tree(tree, word, meaning):
    '''
    We build word search trees, where we walk down
    the letters of a word. For example:
      你 Good
      你好 Hello
    Would build the tree
         你
        /  \
      You   好
             \
           Hello
    '''
    if len(word) == 0:
        tree[''] = meaning
    else:
        _add_to_tree(tree[word[0]], word[1:], meaning)
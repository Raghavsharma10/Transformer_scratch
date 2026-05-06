def select(self, model):
    """Select nodes according to the input selector.

    This can ALWAYS return multiple root elements.
    """
    res = []

    def doSelect(value, pre, remaining):
      if not remaining:
        res.append((pre, value))
      else:
        # For the other selectors to work, value must be a Tuple or a list at this point.
        if not is_tuple(value) and not isinstance(value, list):
          return

        qhead, qtail = remaining[0], remaining[1:]
        if isinstance(qhead, tuple) and is_tuple(value):
          for alt in qhead:
            if alt in value:
              doSelect(value[alt], pre + [alt], qtail)
        elif qhead == '*':
          if isinstance(value, list):
            indices = range(len(value))
            reprs = [listKey(i) for i in indices]
          else:
            indices = value.keys()
            reprs = indices

          for key, rep in zip(indices, reprs):
            doSelect(value[key], pre + [rep], qtail)
        elif isinstance(qhead, int) and isinstance(value, list):
          doSelect(value[qhead], pre + [listKey(qhead)], qtail)
        elif is_tuple(value):
          if qhead in value:
            doSelect(value[qhead], pre + [qhead], qtail)

    for selector in self.selectors:
      doSelect(model, [], selector)

    return QueryResult(res)
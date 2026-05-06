def addChild(self,item):
    """
    When you add a child to a Node, you are adding yourself as a parent to the child
    You cannot have the same node as a child more than once.
    If you add a Node, it is used. If you add a non-node, a new child Node is created. Thus: You cannot
    add a child as an item which is a Node. (You can, however, construct such a node, and add it as a child)
    """
    if not isinstance(item,Node):
      item = Node(item)
    if item in self.children:
      return item
    self.children.append(item)
    item.parents.add(self)
    return item
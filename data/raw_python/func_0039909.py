def makeDependencyMap(aMap):
  """
  create a dependency data structure as follows:
  - Each key in aMap represents an item that depends on each item in the iterable which is that key's value
  - Each Node represents an item which is a precursor to its parents and depends on its children
  Returns a map whose keys are the items described in aMap and whose values are the dependency (sub)tree for that item
  Thus, for aMap = {a:(b,c), b:(d,), c:[]},
  returns {a:Node(a),b:Node(b),c:Node(c),d:Node(d)} where
     - Node(a) has no parent and children: Node(b) and Node(c)
     - Node(b) has parent: Node(a) and child: Node(d)
     - Node(c) has parent: Node(a) and no child
     - Node(d) which was not a key in aMap was created. It has parent: Node(b) and no child
  This map is used to find the precursors for a given item by using BottomUpVisitor on the Node associated with that item
  """
  index = {}
  for i in aMap.keys():
    iNode = index.get(i,None)
    if not iNode:
      iNode = Node(i)
      index[i] = iNode
    for c in aMap[i]:
      cNode = index.get(c,None)
      if not cNode:
        cNode = Node(c)
        index[c] = cNode
      iNode.addChild(cNode)
  return index
def dependencyOrder(aMap, aList = None):
  """
  Given descriptions of dependencies in aMap and an optional list of items in aList
  if not aList, aList = aMap.keys()
  Returns a list containing each element of aList and all its precursors so that every precursor of
  any element in the returned list is seen before that dependent element.
  If aMap contains cycles, something will happen. It may not be pretty...
  """
  dependencyMap = makeDependencyMap(aMap)
  outputList = []
  if not aList:
    aList = aMap.keys()
  items = []
  v = BottomUpVisitor()
  for item in aList:
    try:
      v.visit(dependencyMap[item])
    except KeyError:
      outputList.append(item)
  outputList = [x.item for x in v.history]+outputList
  return outputList
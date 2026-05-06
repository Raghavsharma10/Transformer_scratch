def itemTypeWithSomeAttributes(attributeTypes):
    """
    Create a new L{Item} subclass with L{numAttributes} integers in its
    schema.
    """
    class SomeItem(Item):
        typeName = 'someitem_' + str(typeNameCounter())
        for i, attributeType in enumerate(attributeTypes):
            locals()['attr_' + str(i)] = attributeType()
    return SomeItem
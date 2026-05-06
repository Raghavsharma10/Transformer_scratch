def dependsOn(itemType, itemCustomizer=None, doc='',
              indexed=True, whenDeleted=reference.NULLIFY):
    """
    This function behaves like L{axiom.attributes.reference} but with
    an extra behaviour: when this item is installed (via
    L{axiom.dependency.installOn} on a target item, the
    type named here will be instantiated and installed on the target
    as well.

    For example::

      class Foo(Item):
          counter = integer()
          thingIDependOn = dependsOn(Baz, lambda baz: baz.setup())

    @param itemType: The Item class to instantiate and install.
    @param itemCustomizer: A callable that accepts the item installed
    as a dependency as its first argument. It will be called only if
    an item is created to satisfy this dependency.

    @return: An L{axiom.attributes.reference} instance.
    """

    frame = sys._getframe(1)
    locals = frame.f_locals

    # Try to make sure we were called from a class def.
    if (locals is frame.f_globals) or ('__module__' not in locals):
        raise TypeError("dependsOn can be used only from a class definition.")
    ref = reference(reftype=itemType, doc=doc, indexed=indexed, allowNone=True,
                    whenDeleted=whenDeleted)
    if "__dependsOn_advice_data__" not in locals:
        addClassAdvisor(_dependsOn_advice)
    locals.setdefault('__dependsOn_advice_data__', []).append(
    (itemType, itemCustomizer, ref))
    return ref
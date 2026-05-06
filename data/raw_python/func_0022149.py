def _subclass_container(self):
        """
        Call subclasses via function to allow passing parent namespace to subclasses.

        **Returns:** dict with subclass references.
        """
        _parent_class = self

        class GetWrapper(Get):

            def __init__(self):
                self._parent_class = _parent_class

        class PostWrapper(Post):

            def __init__(self):
                self._parent_class = _parent_class

        class PutWrapper(Put):

            def __init__(self):
                self._parent_class = _parent_class

        class PatchWrapper(Patch):

            def __init__(self):
                self._parent_class = _parent_class

        class DeleteWrapper(Delete):

            def __init__(self):
                self._parent_class = _parent_class

        class InteractiveWrapper(Interactive):

            def __init__(self):
                self._parent_class = _parent_class

        return {"get": GetWrapper,
                "post": PostWrapper,
                "put": PutWrapper,
                "patch": PatchWrapper,
                "delete": DeleteWrapper,
                "interactive": InteractiveWrapper}
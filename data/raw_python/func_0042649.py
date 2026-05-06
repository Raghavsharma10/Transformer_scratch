def get_preview_kwargs(self, **kwargs):
        """
        Gets the url keyword arguments to pass to the
        `preview_view` callable. If the `pass_through_kwarg`
        attribute is set the value of `pass_through_attr` will
        be looked up on the object.

        So if you are previewing an item Obj<id=2> and

            ::

                self.pass_through_kwarg = 'object_id'
                self.pass_through_attr = 'pk'

        This will return

            ::

                { 'object_id' : 2 }

        """
        if not self.pass_through_kwarg:
            return {}

        obj = self.get_object()
        return {
            self.pass_through_kwarg: getattr(obj, self.pass_through_attr)
        }
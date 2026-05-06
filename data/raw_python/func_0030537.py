def transform(self, img, transformation, params):
        '''
        Apply transformations to the image.

        New transformations can be defined as methods::

            def do__transformationname(self, img, transformation, params):
                'returns new image with transformation applied'
                ...

            def new_size__transformationname(self, size, target_size, params):
                'dry run, returns a size of image if transformation is applied'
                ...
        '''
        # Transformations MUST be idempotent.
        # The limitation is caused by implementation of
        # image upload in iktomi.cms.
        # The transformation can be applied twice:
        # on image upload after crop (when TransientFile is created)
        # and on object save (when PersistentFile is created).
        method = getattr(self, 'do__' + transformation)
        return method(img, transformation, params)
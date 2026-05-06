def delete(self):
        """ Delete the object from GPU memory. 

        Note that the GPU object will also be deleted when this gloo
        object is about to be deleted. However, sometimes you want to
        explicitly delete the GPU object explicitly.
        """
        # We only allow the object from being deleted once, otherwise
        # we might be deleting another GPU object that got our gl-id
        # after our GPU object was deleted. Also note that e.g.
        # DataBufferView does not have the _glir attribute.
        if hasattr(self, '_glir'):
            # Send our final command into the queue
            self._glir.command('DELETE', self._id)
            # Tell master glir queue that this queue is no longer being used
            self._glir._deletable = True
            # Detach the queue
            del self._glir
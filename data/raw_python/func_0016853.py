def _compute(self, feed_dict, shard):
        """ Call the tensorflow compute """

        try:
            descriptor, enq = self._tfrun(self._tf_expr[shard], feed_dict=feed_dict)
            self._inputs_waiting.decrement(shard)

        except Exception as e:
            montblanc.log.exception("Compute Exception")
            raise
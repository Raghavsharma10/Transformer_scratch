def _poll_worker_result(self, group_id):
        """
        从队列里面获取worker的返回
        """
        while 1:
            try:
                msg = self.parent_input_dict[group_id].get()
            except KeyboardInterrupt:
                break
            except:
                logger.error('exc occur.', exc_info=True)
                break

            # 参考 http://twistedsphinx.funsize.net/projects/core/howto/threading.html
            reactor.callFromThread(self._handle_worker_response, msg)
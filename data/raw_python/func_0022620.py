def _tmgr(self, uid, rmgr, logger, mq_hostname, port, pending_queue, completed_queue):
        """
        **Purpose**: Method to be run by the tmgr process. This method receives a Task from the pending_queue
        and submits it to the RTS. At all state transititons, they are synced (blocking) with the AppManager
        in the master process.

        In addition, the tmgr also receives heartbeat 'request' msgs from the heartbeat-req queue. It responds with a
        'response' message to the 'heartbeart-res' queue.

        **Details**: The AppManager can re-invoke the tmgr process with this function if the execution of the workflow is 
        still incomplete. There is also population of a dictionary, placeholder_dict, which stores the path of each of
        the tasks on the remote machine. 
        """

        raise NotImplementedError('_tmgr() method ' +
                                  'not implemented in TaskManager for %s' % self._rts)
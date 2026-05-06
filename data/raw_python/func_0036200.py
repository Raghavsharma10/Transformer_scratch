def _loop(self, reader):
    """Main execution loop of the scheduler.

    The loop runs every second. Between iterations, the loop listens for
    schedule or cancel requests coming from Flask via over the gipc pipe
    (reader) and modifies the queue accordingly.

    When a task completes, it is rescheduled
    """
    results = set()

    while True:
      now = datetime.datetime.now()
      if self._task_queue and self._task_queue[0][0] <= now:
        task = heappop(self._task_queue)[1]
        if task['id'] not in self._pending_cancels:
          result = self._executor.submit(_execute, task)
          results.add(result)
        else:
          self._pending_cancels.remove(task['id'])
      else:
        # Check for new tasks coming from HTTP
        with gevent.Timeout(0.5, False) as t:
          message = reader.get(timeout=t)
          if message[0] == 'schedule':
            self._schedule(message[1], next_run=now)
          elif message[0] == 'cancel':
            self._cancel(message[1])
        # Reschedule completed tasks
        if not results:
          gevent.sleep(0.5)
          continue
        ready = self._executor.wait(results, num=1, timeout=0.5)
        for result in ready:
          results.remove(result)
          if result.value:
            task = result.value
            interval = int(task['interval'])
            if interval:
              run_at = now + datetime.timedelta(seconds=int(task['interval']))
              self._schedule(task, next_run=run_at)
          else:
            err_msg = result.exception
            sys.stderr.write("ERROR: %s" % err_msg)
            email_msg = 'Task %s failed at %s\n\n%s' % (
              task['id'],
              datetime.datetime.now(),
              err_msg
            )
            send_mail(get_app().config['SCHEDULER_FAILURE_EMAILS'],
                      'Scheduler Failure',
                      email_msg)
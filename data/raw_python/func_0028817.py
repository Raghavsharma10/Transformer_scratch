def wait_for_prepare_to_finish(
            self,
            prepare_id,
            sec_to_sleep=5.0,
            max_retries=100000):
        """wait_for_prepare_to_finish

        :param prepare_id: MLPrepare.id to wait on
        :param sec_to_sleep: seconds to sleep during polling
        :param max_retries: max retires until stopping
        """

        not_done = True
        retry_attempt = 1
        while not_done:

            if self.debug:
                log.info(("PREPSTATUS getting prepare.id={} details")
                         .format(
                            prepare_id))

            response = self.get_prepare_by_id(prepare_id)

            if self.debug:
                log.info(("PREPSTATUS got prepare.id={} response={}")
                         .format(
                            prepare_id,
                            response))

            if response["status"] != SUCCESS:
                log.error(("PREPSTATUS failed to get prepare.id={} "
                           "with error={}")
                          .format(
                            prepare_id,
                            response["error"]))
                return self.build_response(
                    status=ERROR,
                    error=response["error"],
                    data=response["data"])
            # stop if this failed getting the prepare details

            prepare_data = response.get(
                "data",
                None)

            if not prepare_data:
                return self.build_response(
                    status=ERROR,
                    error="failed to find prepare dictionary in response",
                    data=response["data"])

            prepare_status = prepare_data["status"]

            if prepare_status == "finished" \
               or prepare_status == "completed":

                not_done = False
                return self.build_response(
                    status=SUCCESS,
                    error="",
                    data=prepare_data)
            else:

                retry_attempt += 1
                if retry_attempt > max_retries:
                    err_msg = ("failed waiting "
                               "for prepare.id={} to finish").format(
                                   prepare_id)
                    log.error(err_msg)
                    return self.build_response(
                        status=ERROR,
                        error=err_msg)
                else:
                    if self.verbose:
                        if retry_attempt % 100 == 0:
                            log.info(("waiting on prepare.id={} retry={}")
                                     .format(
                                        prepare_id,
                                        retry_attempt))
                    # if logging just to show this is running
                    time.sleep(sec_to_sleep)
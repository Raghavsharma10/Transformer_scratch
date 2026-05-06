def wait_for_job_to_finish(
            self,
            job_id,
            sec_to_sleep=5.0,
            max_retries=100000):
        """wait_for_job_to_finish

        :param job_id: MLJob.id to wait on
        :param sec_to_sleep: seconds to sleep during polling
        :param max_retries: max retires until stopping
        """

        not_done = True
        retry_attempt = 1
        while not_done:

            if self.debug:
                log.info(("JOBSTATUS getting job.id={} details")
                         .format(
                            job_id))

            response = self.get_job_by_id(job_id)

            if self.debug:
                log.info(("JOBSTATUS got job.id={} response={}")
                         .format(
                            job_id,
                            response))

            if response["status"] != SUCCESS:
                log.error(("JOBSTATUS failed to get job.id={} with error={}")
                          .format(
                            job_id,
                            response["error"]))
                return self.build_response(
                    status=ERROR,
                    error=response["error"],
                    data=response["data"])
            # stop if this failed getting the job details

            job_data = response.get(
                "data",
                None)

            if not job_data:
                return self.build_response(
                    status=ERROR,
                    error="failed to find job dictionary in response",
                    data=response["data"])

            job_status = job_data["status"]

            if job_status == "finished" \
               or job_status == "completed" \
               or job_status == "launched":

                if self.debug:
                    log.info(("job.id={} is done with status={}")
                             .format(
                                job_id,
                                job_status))

                result_id = job_data["predict_manifest"]["result_id"]

                if self.debug:
                    log.info(("JOBRESULT getting result.id={} details")
                             .format(
                                result_id))

                response = self.get_result_by_id(result_id)

                if self.debug:
                    log.info(("JOBRESULT got result.id={} response={}")
                             .format(
                                result_id,
                                response))

                if response["status"] != SUCCESS:
                    log.error(("JOBRESULT failed to get "
                               "result.id={} with error={}")
                              .format(
                                result_id,
                                response["error"]))
                    return self.build_response(
                        status=ERROR,
                        error=response["error"],
                        data=response["data"])
                # stop if this failed getting the result details

                result_data = response.get(
                    "data",
                    None)

                if result_data["status"] == "finished":
                    full_response = {
                        "job": job_data,
                        "result": result_data
                    }
                    not_done = False
                    return self.build_response(
                        status=SUCCESS,
                        error="",
                        data=full_response)
                else:
                    if retry_attempt % 100 == 0:
                        if self.verbose:
                            log.info(("result_id={} are not done retry={}")
                                     .format(
                                        result_id,
                                        retry_attempt))

                    retry_attempt += 1
                    if retry_attempt > max_retries:
                        err_msg = ("failed waiting "
                                   "for job.id={} result.id={} "
                                   "to finish").format(
                                    job_id,
                                    result_id)
                        log.error(err_msg)
                        return self.build_response(
                            status=ERROR,
                            error=err_msg)
                    else:
                        time.sleep(sec_to_sleep)
                    # wait while results are written to the db
            else:

                retry_attempt += 1
                if retry_attempt > max_retries:
                    err_msg = ("failed waiting "
                               "for job.id={} to finish").format(
                                   job_id)
                    log.error(err_msg)
                    return self.build_response(
                        status=ERROR,
                        error=err_msg)
                else:
                    if self.verbose:
                        if retry_attempt % 100 == 0:
                            log.info(("waiting on job.id={} retry={}")
                                     .format(
                                        job_id,
                                        retry_attempt))
                    # if logging just to show this is running
                    time.sleep(sec_to_sleep)
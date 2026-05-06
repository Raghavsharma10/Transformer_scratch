def _write_single_sample(self, sample):
        """
        :type sample: Sample
        """
        bytes = sample.extras.get("responseHeadersSize", 0) + 2 + sample.extras.get("responseBodySize", 0)

        message = sample.error_msg
        if not message:
            message = sample.extras.get("responseMessage")
        if not message:
            for sample in sample.subsamples:
                if sample.error_msg:
                    message = sample.error_msg
                    break
                elif sample.extras.get("responseMessage"):
                    message = sample.extras.get("responseMessage")
                    break
        self.writer.writerow({
            "timeStamp": int(1000 * sample.start_time),
            "elapsed": int(1000 * sample.duration),
            "Latency": 0,  # TODO
            "label": sample.test_case,

            "bytes": bytes,

            "responseCode": sample.extras.get("responseCode"),
            "responseMessage": message,
            "allThreads": self.concurrency,  # TODO: there will be a problem aggregating concurrency for rare samples
            "success": "true" if sample.status == "PASSED" else "false",
        })
        self.out_stream.flush()
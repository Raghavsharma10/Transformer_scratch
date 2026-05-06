def capture_termination_signal(please_stop):
    """
    WILL SIGNAL please_stop WHEN THIS AWS INSTANCE IS DUE FOR SHUTDOWN
    """
    def worker(please_stop):
        seen_problem = False
        while not please_stop:
            request_time = (time.time() - timer.START)/60  # MINUTES
            try:
                response = requests.get("http://169.254.169.254/latest/meta-data/spot/termination-time")
                seen_problem = False
                if response.status_code not in [400, 404]:
                    Log.alert("Shutdown AWS Spot Node {{name}} {{type}}", name=machine_metadata.name, type=machine_metadata.aws_instance_type)
                    please_stop.go()
            except Exception as e:
                e = Except.wrap(e)
                if "Failed to establish a new connection: [Errno 10060]" in e or "A socket operation was attempted to an unreachable network" in e:
                    Log.note("AWS Spot Detection has shutdown, probably not a spot node, (http://169.254.169.254 is unreachable)")
                    return
                elif seen_problem:
                    # IGNORE THE FIRST PROBLEM
                    Log.warning("AWS shutdown detection has more than one consecutive problem: (last request {{time|round(1)}} minutes since startup)", time=request_time, cause=e)
                seen_problem = True

                (Till(seconds=61) | please_stop).wait()
            (Till(seconds=11) | please_stop).wait()

    Thread.run("listen for termination", worker)
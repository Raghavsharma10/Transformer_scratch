def run_main(
        need_response=False,
        callback=None):
    """run_main

    start the packet consumers and the packet processors

    :param need_response: should send response back to publisher
    :param callback: handler method
    """

    stop_file = ev("STOP_FILE",
                   "/opt/stop_recording")

    num_workers = int(ev("NUM_WORKERS",
                         "1"))
    shutdown_msg = "SHUTDOWN"

    log.info("Start - {}".format(name))

    log.info("Creating multiprocessing queue")
    tasks = multiprocessing.JoinableQueue()
    queue_to_consume = multiprocessing.Queue()
    host = "localhost"

    # Start consumers
    log.info("Starting Consumers to process queued tasks")
    consumers = start_consumers_for_queue(
        num_workers=num_workers,
        tasks=tasks,
        queue_to_consume=queue_to_consume,
        shutdown_msg=shutdown_msg,
        consumer_class=WorkerToProcessPackets,
        callback=callback)

    log.info("creating socket")
    skt = create_layer_2_socket()
    log.info("socket created")

    not_done = True
    while not_done:

        if not skt:
            log.info("Failed to create layer 2 socket")
            log.info("Please make sure to run as root")
            not_done = False
            break

        try:
            if os.path.exists(stop_file):
                log.info(("Detected stop_file={}")
                         .format(stop_file))
                not_done = False
                break
            # stop if the file exists

            # Only works on linux
            packet = skt.recvfrom(65565)

            if os.path.exists(stop_file):
                log.info(("Detected stop_file={}")
                         .format(stop_file))
                not_done = False
                break
            # stop if the file was created during a wait loop

            tasks.put(NetworkPacketTask(source=host,
                                        payload=packet))

        except KeyboardInterrupt as k:
            log.info("Stopping")
            not_done = False
            break
        except Exception as e:
            log.error(("Failed reading socket with ex={}")
                      .format(e))
            not_done = False
            break
        # end of try/ex during socket receving

    # end of while processing network packets

    log.info(("Shutting down consumers={}")
             .format(len(consumers)))

    shutdown_consumers(num_workers=num_workers,
                       tasks=tasks)

    # Wait for all of the tasks to finish
    if need_response:
        log.info("Waiting for tasks to finish")
        tasks.join()

    log.info("Done waiting for tasks to finish")
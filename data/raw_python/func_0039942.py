def run_spy(group, port, verbose):
    """
    Runs the multicast spy

    :param group: Multicast group
    :param port: Multicast port
    :param verbose: If True, prints more details
    """
    # Create the socket
    socket, group = multicast.create_multicast_socket(group, port)
    print("Socket created:", group, "port:", port)

    # Set the socket as non-blocking
    socket.setblocking(0)

    # Prepare stats storage
    stats = {
        "total_bytes": 0,
        "total_count": 0,
        "sender_bytes": {},
        "sender_count": {},
    }

    print("Press Ctrl+C to exit")
    try:
        loop_nb = 0
        while True:
            if loop_nb % 50 == 0:
                loop_nb = 0
                print("Reading...")

            loop_nb += 1

            ready = select.select([socket], [], [], .1)
            if ready[0]:
                # Socket is ready
                data, sender = socket.recvfrom(1024)
                len_data = len(data)

                # Store stats
                stats["total_bytes"] += len_data
                stats["total_count"] += 1

                try:
                    stats["sender_bytes"][sender] += len_data
                    stats["sender_count"][sender] += 1
                except KeyError:
                    stats["sender_bytes"][sender] = len_data
                    stats["sender_count"][sender] = 1

                print("Got", len_data, "bytes from", sender[0], "port",
                      sender[1], "at", datetime.datetime.now())
                if verbose:
                    print(hexdump(data))
    except KeyboardInterrupt:
        # Interrupt
        print("Ctrl+C received: bye !")

    # Print statistics
    print("Total number of packets:", stats["total_count"])
    print("Total read bytes.......:", stats["total_bytes"])

    for sender in stats["sender_count"]:
        print("\nSender", sender[0], "from port", sender[1])
        print("\tTotal packets:", stats["sender_count"][sender])
        print("\tTotal bytes..:", stats["sender_bytes"][sender])

    return 0
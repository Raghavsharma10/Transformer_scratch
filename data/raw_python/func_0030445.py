def _log_rate(output_f, d, message=None):
    """Log a message for the Nth time the method is called.

    d is the object returned from init_log_rate

    """

    if d[2] <= 0:

        if message is None:
            message = d[4]

        # Average the rate over the length of the deque.
        d[6].append(int(d[3] / (time() - d[1])))
        rate = sum(d[6]) / len(d[6])

        # Prints the processing rate in 1,000 records per sec.
        output_f(message + ': ' + str(rate) + '/s ' + str(d[0] / 1000) + 'K ')

        d[1] = time()

        # If the print_rate was specified, adjust the number of records to
        # aproximate that rate.
        if d[5]:
            target_rate = rate * d[5]
            d[3] = int((target_rate + d[3]) / 2)

        d[2] = d[3]

    d[0] += 1
    d[2] -= 1
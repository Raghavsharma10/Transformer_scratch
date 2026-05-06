def show_help(name):
    """
    Show help and basic usage
    """
    print('Usage: python3 {} [OPTIONS]... '.format(name))
    print('ISO8583 message client')
    print('  -v, --verbose\t\tRun transactions verbosely')
    print('  -p, --port=[PORT]\t\tTCP port to connect to, 1337 by default')
    print('  -s, --server=[IP]\t\tIP of the ISO host to connect to, 127.0.0.1 by default')
    print('  -t, --terminal=[ID]\t\tTerminal ID (used in DE 41 ISO field, 10001337 by default)')
    print('  -m, --merchant=[ID]\t\tMerchant ID (used in DE 42 ISO field, 999999999999001 by default)')
    print('  -k, --terminal-key=[KEY]\t\tTerminal key (\'DEADBEEF DEADBEEF DEADBEEF DEADBEEF\' by default)')
    print('  -K, --master-key=[KEY]\t\Master key (\'ABABABAB CDCDCDCD EFEFEFEF AEAEAEAE\' by default)')
    print('  -f, --file=[file.xml]\t\tUse transaction data from the given XML-file')
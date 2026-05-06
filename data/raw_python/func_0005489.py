def send_to_azure_multi_threads(instance, data, nb_threads=4, replace=True, types=None, primary_key=(),
                                sub_commit=False):
    """
    data = {
        "table_name" 	: 'name_of_the_azure_schema' + '.' + 'name_of_the_azure_table' #Must already exist,
        "columns_name" 	: [first_column_name,second_column_name,...,last_column_name],
        "rows"		: [[first_raw_value,second_raw_value,...,last_raw_value],...]
    }
    """

    # Time initialization
    start = datetime.datetime.now()

    # Extract info
    table_name = data["table_name"]
    columns_name = data["columns_name"]
    rows = data["rows"]
    total_len_data = len(rows)

    # Create table if needed
    if not existing_test(instance, table_name) or (types is not None) or (primary_key != ()):
        create.create_table(instance, data, primary_key, types)

    # Clean table if needed
    if replace:
        cleaning_function(instance, table_name)

    # Define batch size
    batch_size = int(total_len_data / nb_threads) + 1
    if total_len_data < nb_threads:
        batch_size = 1

    # Get table info
    table_info = get_table_info(instance, table_name)

    # Split data in batches of batch_size length
    split_data = []

    # global threads_state
    # threads_state = {}

    for i in range(nb_threads):
        batch = create_a_batch(rows, batch_size, i)
        split_data.append(
            {
                "data":
                    {
                        "table_name": table_name,
                        "columns_name": columns_name,
                        "rows": batch
                    },
                "instance": instance,
                "thread_number": i,
                "nb_threads": nb_threads,
                "sub_commit": sub_commit,
                "table_info": table_info,
            }
        )
        write_in_file("threads_state_%s" % str(i), str({
            "iteration": 0,
            "total": len(batch)
        }))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        r = list(executor.map(send_to_azure_from_one_thread, split_data))

    print()
    for num_thread in range(nb_threads):
        insert_query = "INSERT INTO %s SELECT * FROM %s" % (table_name, table_name + "_" + str(num_thread))
        print(insert_query)
        execute_query(instance, insert_query)

    for num_thread in range(nb_threads):
        sub_table = table_name + "_" + str(num_thread)
        print(C.HEADER + "DROP TABLE %s..." % sub_table + C.ENDC)
        execute_query(instance, "DROP TABLE %s" % sub_table)
        print(C.HEADER + "DROP TABLE %s...OK" % sub_table + C.ENDC)

    total_length_data = 0
    for element in split_data:
        total_length_data = total_length_data + len(element["data"]["rows"])

    for i in range(len(r)):
        print("Thread %s : %s seconds" % (str(i), str(r[i])))

    print("Total rows: %s" % str(total_length_data))
    print(C.BOLD + "Total time in seconds : %s" % str((datetime.datetime.now() - start).seconds) + C.ENDC)
    return 0
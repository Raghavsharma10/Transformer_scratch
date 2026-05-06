def send_to_azure(instance, data, thread_number, sub_commit, table_info, nb_threads):
    """
    data = {
        "table_name" 	: 'name_of_the_azure_schema' + '.' + 'name_of_the_azure_table' #Must already exist,
        "columns_name" 	: [first_column_name,second_column_name,...,last_column_name],
        "rows"		: [[first_raw_value,second_raw_value,...,last_raw_value],...]
    }
    """

    rows = data["rows"]
    if not rows:
        return 0
    columns_name = data["columns_name"]
    table_name = data["table_name"] + "_" + str(thread_number)

    print(C.HEADER + "Create table %s..." % table_name + C.ENDC)
    create_table_from_info(instance, table_info, table_name)
    print(C.OKGREEN + "Create table %s...OK" % table_name + C.ENDC)
    small_batch_size = int(2099 / len(columns_name))

    cnxn = connect(instance)
    cursor = cnxn.cursor()

    # Initialize counters
    boolean = True
    total_rows = len(rows)
    question_mark_pattern = "(%s)" % ",".join(["?" for i in range(len(rows[0]))])
    counter = 0
    while boolean:
        temp_row = []
        question_mark_list = []
        for i in range(small_batch_size):
            if rows:
                temp_row.append(rows.pop())
                question_mark_list.append(question_mark_pattern)
            else:
                boolean = False
                continue
        counter = counter + len(temp_row)
        # percent = round(float(counter * 100) / total_rows)
        threads_state = eval(read_file("threads_state_%s" % str(thread_number)))
        threads_state["iteration"] = counter
        write_in_file("threads_state_%s" % str(thread_number), str(threads_state))

        # print(threads_state)
        if sub_commit:
            suffix = "rows sent"
            # print("Thread %s : %s %% rows sent" % (str(thread_number), str(percent)))
        else:
            suffix = "rows prepared to be sent"
        print_progress_bar_multi_threads(nb_threads, suffix=suffix)
        # print("Thread %s : %s %% rows prepared to be sent" % (str(thread_number), str(percent)))
        data_values_str = ','.join(question_mark_list)
        columns_name_str = ", ".join(columns_name)
        inserting_request = '''INSERT INTO %s (%s) VALUES %s ;''' % (table_name, columns_name_str, data_values_str)

        final_data = [y for x in temp_row for y in x]
        if final_data:
            cursor.execute(inserting_request, final_data)

        if sub_commit:
            commit_function(cnxn)
    if not sub_commit:
        commit_function(cnxn)
    cursor.close()
    cnxn.close()
    return 0
def bag_to_dataframe(bag_name, include=None, exclude=None, parse_header=False, seconds=False):
    '''
    Read in a rosbag file and create a pandas data frame that
    is indexed by the time the message was recorded in the bag.

    :bag_name: String name for the bag file
    :include: None, String, or List  Topics to include in the dataframe
               if None all topics added, if string it is used as regular
                   expression, if list that list is used.
    :exclude: None, String, or List  Topics to be removed from those added
            using the include option using set difference.  If None no topics
            removed. If String it is treated as a regular expression. A list
            removes those in the list.

    :seconds: time index is in seconds

    :returns: a pandas dataframe object
    '''
    # get list of topics to parse
    yaml_info = get_bag_info(bag_name)
    bag_topics = get_topics(yaml_info)
    bag_topics = prune_topics(bag_topics, include, exclude)
    length = get_length(bag_topics, yaml_info)
    msgs_to_read, msg_type = get_msg_info(yaml_info, bag_topics, parse_header)

    bag = rosbag.Bag(bag_name)
    dmap = create_data_map(msgs_to_read)

    # create datastore
    datastore = {}
    for topic in dmap.keys():
        for f, key in dmap[topic].iteritems():
            t = msg_type[topic][f]
            if isinstance(t, int) or isinstance(t, float):
                arr = np.empty(length)
                arr.fill(np.NAN)
            elif isinstance(t, list):
                arr = np.empty(length)
                arr.fill(np.NAN)
                for i in range(len(t)):
                    key_i = '{0}{1}'.format(key, i)
                    datastore[key_i] = arr.copy()
                continue
            else:
                arr = np.empty(length, dtype=np.object)
            datastore[key] = arr

    # create the index
    index = np.empty(length)
    index.fill(np.NAN)

    # all of the data is loaded
    for idx, (topic, msg, mt) in enumerate(bag.read_messages(topics=bag_topics)):
        try:
            if seconds:
                index[idx] = msg.header.stamp.to_sec()
            else:
                index[idx] = msg.header.stamp.to_nsec()
        except:
            if seconds:
                index[idx] = mt.to_sec()
            else:
                index[idx] = mt.to_nsec()
        fields = dmap[topic]
        for f, key in fields.iteritems():
            try:
                d = get_message_data(msg, f)
                if isinstance(d, tuple):
                    for i, val in enumerate(d):
                        key_i = '{0}{1}'.format(key, i)
                        datastore[key_i][idx] = val
                else:
                    datastore[key][idx] = d
            except:
                pass

    bag.close()

    # convert the index
    if not seconds:
        index = pd.to_datetime(index, unit='ns')

    # now we have read all of the messages its time to assemble the dataframe
    return pd.DataFrame(data=datastore, index=index)
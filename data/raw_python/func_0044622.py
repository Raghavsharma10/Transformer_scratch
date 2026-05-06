def process_results(qry_results):
    """Generate dictionary of results from query.

    Decodes the large dict recturned from the AWS query.

    Args:
        qry_results (dict): results from awsc.get_inst_info
    Returns:
        i_info (dict): information on instances and details.

    """
    i_info = {}
    for i, j in enumerate(qry_results['Reservations']):
        i_info[i] = {'id': j['Instances'][0]['InstanceId']}
        i_info[i]['state'] = j['Instances'][0]['State']['Name']
        i_info[i]['ami'] = j['Instances'][0]['ImageId']
        i_info[i]['ssh_key'] = j['Instances'][0]['KeyName']
        i_info[i]['pub_dns_name'] = j['Instances'][0]['PublicDnsName']
        try:
            i_info[i]['tag'] = process_tags(j['Instances'][0]['Tags'])
        except KeyError:
            i_info[i]['tag'] = {"Name": ""}
    debg.dprint("numInstances: ", len(i_info))
    debg.dprintx("Details except AMI-name")
    debg.dprintx(i_info, True)
    return i_info
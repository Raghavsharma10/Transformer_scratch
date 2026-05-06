def rewrite_elife_funding_awards(json_content, doi):
    """ rewrite elife funding awards """

    # remove a funding award
    if doi == "10.7554/eLife.00801":
        for i, award in enumerate(json_content):
            if "id" in award and award["id"] == "par-2":
                del json_content[i]

    # add funding award recipient
    if doi == "10.7554/eLife.04250":
        recipients_for_04250 = [{"type": "person", "name": {"preferred": "Eric Jonas", "index": "Jonas, Eric"}}]
        for i, award in enumerate(json_content):
            if "id" in award and award["id"] in ["par-2", "par-3", "par-4"]:
                if "recipients" not in award:
                    json_content[i]["recipients"] = recipients_for_04250

    # add funding award recipient
    if doi == "10.7554/eLife.06412":
        recipients_for_06412 = [{"type": "person", "name": {"preferred": "Adam J Granger", "index": "Granger, Adam J"}}]
        for i, award in enumerate(json_content):
            if "id" in award and award["id"] == "par-1":
                if "recipients" not in award:
                    json_content[i]["recipients"] = recipients_for_06412

    return json_content
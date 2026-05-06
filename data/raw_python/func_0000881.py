def rewrite_elife_authors_json(json_content, doi):
    """ this does the work of rewriting elife authors json """

    # Convert doi from testing doi if applicable
    article_doi = elifetools.utils.convert_testing_doi(doi)

    # Edge case fix an affiliation name
    if article_doi == "10.7554/eLife.06956":
        for i, ref in enumerate(json_content):
            if ref.get("orcid") and ref.get("orcid") == "0000-0001-6798-0064":
                json_content[i]["affiliations"][0]["name"] = ["Cambridge"]

    # Edge case fix an ORCID
    if article_doi == "10.7554/eLife.09376":
        for i, ref in enumerate(json_content):
            if ref.get("orcid") and ref.get("orcid") == "000-0001-7224-925X":
                json_content[i]["orcid"] = "0000-0001-7224-925X"

    # Edge case competing interests
    if article_doi == "10.7554/eLife.00102":
        for i, ref in enumerate(json_content):
            if not ref.get("competingInterests"):
                if ref["name"]["index"].startswith("Chen,"):
                    json_content[i]["competingInterests"] = "ZJC: Reviewing Editor, <i>eLife</i>"
                elif ref["name"]["index"].startswith("Li,"):
                    json_content[i]["competingInterests"] = "The remaining authors have no competing interests to declare."
    if article_doi == "10.7554/eLife.00270":
        for i, ref in enumerate(json_content):
            if not ref.get("competingInterests"):
                if ref["name"]["index"].startswith("Patterson,"):
                    json_content[i]["competingInterests"] = "MP: Managing Executive Editor, <i>eLife</i>"

    # Remainder of competing interests rewrites
    elife_author_competing_interests = {}
    elife_author_competing_interests["10.7554/eLife.00133"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.00190"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.00230"] = "The authors have declared that no competing interests exist"
    elife_author_competing_interests["10.7554/eLife.00288"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.00352"] = "The author declares that no competing interest exist"
    elife_author_competing_interests["10.7554/eLife.00362"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.00475"] = "The remaining authors have no competing interests to declare."
    elife_author_competing_interests["10.7554/eLife.00592"] = "The other authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.00633"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.02725"] = "The other authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.02935"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.04126"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.04878"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.05322"] = "The other authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.06011"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.06416"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.07383"] = "The other authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.08421"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.08494"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.08648"] = "The other authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.08924"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.09083"] = "The other authors declare that no competing interests exists."
    elife_author_competing_interests["10.7554/eLife.09102"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.09460"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.09591"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.09600"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.10113"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.10230"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.10453"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.10635"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.11407"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.11473"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.11750"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.12217"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.12620"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.12724"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.13023"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.13732"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.14116"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.14258"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.14694"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.15085"] = "The other authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.15312"] = "The other authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.16011"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.16940"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.17023"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.17092"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.17218"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.17267"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.17523"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.17556"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.17769"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.17834"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.18101"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.18515"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.18544"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.18648"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.19071"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.19334"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.19510"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.20183"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.20242"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.20375"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.20797"] = "The other authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.21454"] = "The authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.21491"] = "The other authors declare that no competing interests exist."
    elife_author_competing_interests["10.7554/eLife.22187"] = "The authors declare that no competing interests exist."

    if article_doi in elife_author_competing_interests:
        for i, ref in enumerate(json_content):
            if not ref.get("competingInterests"):
                json_content[i]["competingInterests"] = elife_author_competing_interests[article_doi]

    # Rewrite "other authors declare" ... competing interests statements using a string match
    for i, ref in enumerate(json_content):
        if (ref.get("competingInterests") and (
            ref.get("competingInterests").startswith("The other author") or
            ref.get("competingInterests").startswith("The others author") or
            ref.get("competingInterests").startswith("The remaining authors") or
            ref.get("competingInterests").startswith("The remaining have declared")
            )):
            json_content[i]["competingInterests"] = "No competing interests declared."

    return json_content
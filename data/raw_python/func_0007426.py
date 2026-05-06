def main(left_path, left_column, right_path, right_column,
         outfile, titles, join, minscore, count, warp):
    """Perform the similarity join"""
    right_file = csv.reader(open(right_path, 'r'))
    if titles:
        right_header = next(right_file)
    index = NGram((tuple(r) for r in right_file),
                  threshold=minscore,
                  warp=warp, key=lambda x: lowstrip(x[right_column]))
    left_file = csv.reader(open(left_path, 'r'))
    out = csv.writer(open(outfile, 'w'), lineterminator='\n')
    if titles:
        left_header = next(left_file)
        out.writerow(left_header + ["Rank", "Similarity"] + right_header)
    for row in left_file:
        if not row: continue # skip blank lines
        row = tuple(row)
        results = index.search(lowstrip(row[left_column]), threshold=minscore)
        if results:
            if count > 0:
                results = results[:count]
            for rank, result in enumerate(results, 1):
                out.writerow(row + (rank, result[1]) + result[0])
        elif join == "outer":
            out.writerow(row)
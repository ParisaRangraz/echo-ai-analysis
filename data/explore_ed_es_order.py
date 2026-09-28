"""
explore_ed_es_order.py

The pipeline assumed half_sequence always runs ED -> ES. That was verified for
one patient only. Info_*.cfg states the ED and ES frame numbers explicitly, so
this checks how often the assumption actually holds.
"""

from loader import read_patient_list, parse_info_cfg

for split in ("training", "validation"):
    ids = read_patient_list(split)

    ed_first = 0
    es_first = 0
    other = []

    for pid in ids:
        for view in ("4CH", "2CH"):
            info = parse_info_cfg(pid, view)
            ed, es, n = info["ED"], info["ES"], info["NbFrame"]

            # cfg frame numbers are 1-based; frame 1 is index 0.
            if ed == 1 and es == n:
                ed_first += 1
            elif es == 1 and ed == n:
                es_first += 1
            else:
                other.append((pid, view, ed, es, n))

    total = ed_first + es_first + len(other)
    print(f"\n--- {split} ({total} sequences) ---")
    print(f"ED first, ES last: {ed_first}")
    print(f"ES first, ED last: {es_first}")
    print(f"Neither:           {len(other)}")

    for item in other[:5]:
        print(f"   {item[0]} {item[1]}: ED={item[2]}, ES={item[3]}, NbFrame={item[4]}")
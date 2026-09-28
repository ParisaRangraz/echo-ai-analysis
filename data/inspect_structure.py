"""
inspect_structure.py

Run this FIRST, before load_camus.py, to confirm the actual folder/file
layout of your downloaded CAMUS zip. CAMUS's expected structure (per the
official dataset description) is:

    <root>/
        training/
            patient0001/
                patient0001_2CH_ED.mhd
                patient0001_2CH_ED_gt.mhd
                patient0001_2CH_ES.mhd
                patient0001_2CH_ES_gt.mhd
                patient0001_2CH_sequence.mhd
                patient0001_4CH_ED.mhd
                patient0001_4CH_ED_gt.mhd
                patient0001_4CH_ES.mhd
                patient0001_4CH_ES_gt.mhd
                patient0001_4CH_sequence.mhd
                Info_2CH.cfg
                Info_4CH.cfg
            patient0002/
            ...
        testing/
            patient0450/
            ...

.mhd files are paired with a .raw file of the same name (mhd is the header,
raw is the binary pixel data — SimpleITK reads both together via the .mhd).

This script does NOT assume the above is correct — it just reports what's
actually there, so we can fix load_camus.py if your zip differs.
"""

import os
import zipfile
import sys
from collections import Counter

def inspect_zip(zip_path, max_list=20):
    if not os.path.exists(zip_path):
        print(f"File not found: {zip_path}")
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()

    print(f"Total entries in zip: {len(names)}\n")

    # Show the top few levels of the tree
    top_level = sorted(set(n.split("/")[0] for n in names if n))
    print("Top-level entries:", top_level[:max_list])

    # Try to find training/testing folders regardless of nesting depth
    training_like = [n for n in names if "training" in n.lower()]
    testing_like = [n for n in names if "testing" in n.lower()]
    print(f"\nEntries matching 'training': {len(training_like)}")
    print(f"Entries matching 'testing': {len(testing_like)}")

    # Sample a few full paths to see actual nesting/naming
    print("\nSample paths (first 15):")
    for n in names[:15]:
        print(" ", n)

    # Count file extensions
    ext_counts = Counter(os.path.splitext(n)[1] for n in names if not n.endswith("/"))
    print("\nFile extension counts:", dict(ext_counts))

    # Try to find one patient folder and list its contents
    patient_dirs = sorted(set(
        "/".join(n.split("/")[:2]) for n in names
        if "patient" in n.lower()
    ))
    if patient_dirs:
        sample_patient = patient_dirs[0]
        print(f"\nContents of sample patient folder ('{sample_patient}'):")
        for n in names:
            if n.startswith(sample_patient) and not n.endswith("/"):
                print(" ", n)
    else:
        print("\nNo 'patient' folders detected — structure may differ from expected.")

    print("\nDone. Compare this output against the docstring above and let "
          "Claude know if it matches or differs before we build load_camus.py.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_structure.py /path/to/CAMUS.zip")
        sys.exit(1)
    inspect_zip(sys.argv[1])

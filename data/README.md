# Data access

This project uses the CAMUS dataset, which is **not included** in this
repository and is not redistributed in any form.

## How to get it

1. Register a free account at the Human Heart Project database:
   https://humanheart-project.creatis.insa-lyon.fr
2. Download the CAMUS collection (the NIfTI distribution, `CAMUS_public.zip`).
3. Extract it anywhere outside this repository.
4. Point `DATA_ROOT` and `SPLIT_ROOT` in `data/loader.py` at the extracted
   `database_nifti/` and `database_split/` folders.

The official split files in `database_split/` are used as-is, so no manual
train/validation/test division is needed.

## Terms of use

The dataset carries its own licence, separate from this repository's MIT licence.
It permits research use but **prohibits redistribution**, which is why no image,
mask or preprocessed cache from it appears here — `.gitignore` excludes them all.
Anyone using this code must obtain the dataset themselves under those terms.

## Citation

The dataset authors require citation of:

> S. Leclerc, E. Smistad, J. Pedrosa, A. Ostvik, et al., "Deep Learning for Segmentation
> using an Open Large-Scale Dataset in 2D Echocardiography," IEEE Transactions on Medical
> Imaging, vol. 38, no. 9, pp. 2198-2210, Sept. 2019.
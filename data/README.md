# Data access

This project uses the CAMUS dataset, which is **not included** in this
repository and is not redistributed in any form.

## How to get it

1. Register a free account at the Human Heart Project database:
   https://humanheart-project.creatis.insa-lyon.fr
2. Download the CAMUS collection (the NIfTI distribution, `CAMUS_public.zip`).
3. Extract it. It contains a `database_nifti/` folder and a `database_split/`
   folder.
4. Tell the code where it is, in one of two ways:
   - set the `ECHO_DATA_ROOT` environment variable to the folder that contains
     `database_nifti/` and `database_split/` (recommended); or
   - place that folder at `data/CAMUS_public/CAMUS_public/` inside the
     repository, which is the default location the code looks in.

   On Windows PowerShell, setting the variable for the current session looks like:

   ```
   $env:ECHO_DATA_ROOT = "C:\path\to\CAMUS_public\CAMUS_public"
   ```

   No dataset path is stored in the code, so nothing machine-specific needs to
   be edited.

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
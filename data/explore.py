import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import torch

# print("PyTorch version:", torch.__version__)
# print("CUDA available (GPU):", torch.cuda.is_available())
# # # File paths
# # image_path = r"<ECHO_DATA_ROOT>/database_nifti\patient0001\patient0001_4CH_ED.nii.gz"
# # gt_path = r"<ECHO_DATA_ROOT>/database_nifti\patient0001\patient0001_4CH_ED_gt.nii.gz"

# # # Read images
# # img = sitk.ReadImage(image_path)
# # gt_img = sitk.ReadImage(gt_path)

# # # Convert to NumPy Array for further processing and visualization
# # img_arr = sitk.GetArrayFromImage(img)
# # gt_arr = sitk.GetArrayFromImage(gt_img)

# # print("Image Array Shape:", img_arr.shape)
# # print("Ground Truth Array Shape:", gt_arr.shape)
# # print("Pixel Type:", img.GetPixelIDTypeAsString())
# # print("Unique labels in GT:", np.unique(gt_arr))
# # print("Image min/max:", img_arr.min(), img_arr.max())
# # plt.figure(figsize=(10, 5))

# # plt.subplot(1, 2, 1)
# # plt.imshow(img_arr, cmap="gray")
# # plt.title("Original Echo Image")

# # plt.subplot(1, 2, 2)
# # plt.imshow(gt_arr, cmap="viridis")
# # plt.title("Ground Truth Mask")

# # plt.show()



# # # --- Reusable loader function ---
# # DATA_ROOT = r"<ECHO_DATA_ROOT>/database_nifti"

# # def load_patient_frame(patient_id, view="4CH", phase="ED"):
# #     """
# #     Load a single echo frame and its ground-truth mask for one patient.

# #     Args:
# #         patient_id (str): e.g. "patient0001"
# #         view (str): "2CH" or "4CH"
# #         phase (str): "ED" (end-diastole) or "ES" (end-systole)

# #     Returns:
# #         img_arr (np.ndarray): the echo image as a numpy array
# #         gt_arr (np.ndarray): the ground-truth segmentation mask, same shape as img_arr
# #     """
# #     img_path = f"{DATA_ROOT}\\{patient_id}\\{patient_id}_{view}_{phase}.nii.gz"
# #     gt_path = f"{DATA_ROOT}\\{patient_id}\\{patient_id}_{view}_{phase}_gt.nii.gz"

# #     img = sitk.ReadImage(img_path)
# #     gt_img = sitk.ReadImage(gt_path)

# #     img_arr = sitk.GetArrayFromImage(img)
# #     gt_arr = sitk.GetArrayFromImage(gt_img)

# #     return img_arr, gt_arr

# from loader import load_patient_frame, resize_image, read_patient_list
# from preprocess import preprocess_one_patient
# # --- Loop over the first few patients ---
# patient_ids = ["patient0001", "patient0002", "patient0003"]

# for pid in patient_ids:
#     img_arr, gt_arr = load_patient_frame(pid)
#     print(pid, "-> image shape:", img_arr.shape, ", GT shape:", gt_arr.shape)

# # # --- Collect image sizes across the whole training set ---
# # def read_patient_list(txt_path):
# #     """Read a subgroup file (one patient ID per line) into a list."""
# #     with open(txt_path, "r") as f:
# #         ids = [line.strip() for line in f if line.strip()]
# #     return ids

# # training_ids = read_patient_list(
# #     r"<ECHO_DATA_ROOT>/database_split\subgroup_training.txt"
# # )

# # print("Number of training patients:", len(training_ids))
# training_ids = read_patient_list("training")
# print("Number of training patients:", len(training_ids))

# heights = []
# widths = []

# for pid in training_ids:
#     img_arr, _ = load_patient_frame(pid)
#     h, w = img_arr.shape
#     heights.append(h)
#     widths.append(w)

# print("Height range:", min(heights), "-", max(heights))
# print("Width range:", min(widths), "-", max(widths))

# #from skimage.transform import resize as sk_resize

# # def resize_image(arr, target_size=(256, 256), is_mask=False):
# #     """
# #     Resize a 2D numpy array to target_size.
# #     is_mask=True uses nearest-neighbor (order=0) to keep integer class labels intact.
# #     is_mask=False uses linear interpolation (order=1) for smoother real images.
# #     """
# #     order = 0 if is_mask else 1
# #     resized = sk_resize(arr, target_size, order=order, preserve_range=True, anti_aliasing=not is_mask)
# #     return resized.astype(arr.dtype)
# img_arr, gt_arr = load_patient_frame("patient0002")
# img_resized = resize_image(img_arr, target_size=(256, 256), is_mask=False)
# gt_resized = resize_image(gt_arr, target_size=(256, 256), is_mask=True)
# print("After resize:", img_resized.shape, gt_resized.shape)
# print("Unique labels in resized GT:", np.unique(gt_resized))
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img_resized, cmap="gray")
# plt.title("Resized Image (256x256)")

# plt.subplot(1, 2, 2)
# plt.imshow(gt_resized, cmap="viridis")
# plt.title("Resized Mask (256x256)")

# plt.show()

# #test preprocess,normalize
# img, mask = preprocess_one_patient("patient0001")
# print("Image dtype:", img.dtype, ", min/max:", img.min(), img.max())
# print("Mask dtype:", mask.dtype, ", unique values:", np.unique(mask))

# # Test preprocess_split() on the validation set (smaller than training,
# # good for a first test before running on all 400 training patients).
# from preprocess import preprocess_split

# preprocess_split("validation")

# # Now that validation worked correctly, process the full training set too.
# preprocess_split("training")

# # --- Look closely at speckle noise ---
# # Zoom into a small patch of the image to see the grainy speckle pattern
# # that's characteristic of ultrasound (as opposed to smooth camera noise).

# img_arr, gt_arr = load_patient_frame("patient0001")

# plt.figure(figsize=(12, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(img_arr, cmap="gray")
# plt.title("Full Image")

# plt.subplot(1, 2, 2)
# # Crop a small 80x80 patch from somewhere inside the heart region
# patch = img_arr[150:230, 200:280]
# plt.imshow(patch, cmap="gray")
# plt.title("Zoomed Patch (80x80) - see the speckle grain")

# plt.show()

# #denoising
# from loader import load_patient_frame
# from processing import lee_filter
# img_arr, gt_arr = load_patient_frame("patient0001")
# img_normalized = img_arr.astype(np.float32) / 255.0

# filtered = lee_filter(img_normalized, window_size=11)

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(img_normalized, cmap="gray")
# plt.title("Original")
# plt.subplot(1, 2, 2)
# plt.imshow(filtered, cmap="gray")
# plt.title("After Lee Filter")
# plt.show()

# #comparing
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from skimage.metrics import structural_similarity as ssim

# # Compare original vs Lee-filtered (measures how much the filter changed
# # the image, not "closeness to ground truth" since no clean reference exists)
# psnr_value = psnr(img_normalized, filtered, data_range=1.0)
# ssim_value = ssim(img_normalized, filtered, data_range=1.0)

# print(f"PSNR (original vs filtered): {psnr_value:.2f} dB")
# print(f"SSIM (original vs filtered): {ssim_value:.4f}")

# # --- Test the denoiser architecture (no training yet, just shape check) ---
# import sys
# sys.path.append("../models")  # so Python can find denoiser.py in the models/ folder

# from denoiser import SimpleDenoiser

# model = SimpleDenoiser()

# # Create a fake random image, just to test the shapes.
# # Shape: (batch_size, channels, height, width) = (1, 1, 256, 256)
# fake_image = torch.rand(1, 1, 256, 256)

# output = model(fake_image)

# print("Input shape:", fake_image.shape)
# print("Output shape:", output.shape)

# # --- Test the Dataset class ---
# from dataset import CamusDataset

# train_dataset = CamusDataset("processed/training")

# print("Number of samples:", len(train_dataset))

# # Get the first sample
# image, mask = train_dataset[0]
# print("Image tensor shape:", image.shape)
# print("Mask tensor shape:", mask.shape)
# print("Image dtype:", image.dtype)
# print("Image min/max:", image.min().item(), image.max().item())

# # --- Test the DataLoader ---
# from torch.utils.data import DataLoader

# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# # Get one batch
# images, masks = next(iter(train_loader))
# print("Batch of images shape:", images.shape)
# print("Batch of masks shape:", masks.shape)

# from loader import load_half_sequence

# sequence = load_half_sequence("patient0001")
# print("Sequence shape:", sequence.shape)

# # --- Test the U-Net architecture (no training yet, just shape check) ---
# import sys
# sys.path.append("../models")

# from unet import SimpleUNet

# model = SimpleUNet(num_classes=4)

# # Fake random image: (batch_size, channels, height, width)
# fake_image = torch.rand(1, 1, 256, 256)

# output = model(fake_image)

# print("Input shape:", fake_image.shape)
# print("Output shape:", output.shape)
# # --- Check: does half_sequence_gt contain a mask for every frame? ---
# seq_gt = load_half_sequence("patient0001", gt=True)
# print("GT sequence shape:", seq_gt.shape)

# for i in range(seq_gt.shape[0]):
#     labels = np.unique(seq_gt[i])
#     print(f"Frame {i}: labels = {labels}")
# import numpy as np
# from loader import load_half_sequence, load_patient_frame

# sequence = load_half_sequence("patient0001")
# ed_img, _ = load_patient_frame("patient0001", phase="ED")
# es_img, _ = load_patient_frame("patient0001", phase="ES")

# # Are the first/last frames of the sequence identical to the separate ED/ES files?
# print("Frame 0 == ED file: ", np.array_equal(sequence[0], ed_img))
# print("Frame 19 == ES file:", np.array_equal(sequence[-1], es_img))

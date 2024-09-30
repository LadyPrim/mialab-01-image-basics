import numpy as np
import SimpleITK as sitk

import image_basics as ib


def test_load_image():
    """
    TEST_LOAD_IMAGE tests if the load_image function is implemented correctly.
    """
    img_ = ib.load_image("./data/T1native.nii.gz", False)
    load_ok = all(
        (
            isinstance(img_, sitk.Image),
            img_.GetPixelID() == 8,
            img_.GetSize() == (181, 217, 181),
            img_.GetPixel(100, 100, 100) == 12175,
            img_.GetPixel(100, 100, 101) == 11972,
        )
    )
    assert load_ok


def test_to_numpy_array():
    """
    TEST_TO_NUMPY_ARRAY tests if the to_numpy_array function is implemented correctly.
    """
    img_ = ib.load_image("./data/T1native.nii.gz", False)
    np_img_ = ib.to_numpy_array(img_)
    to_numpy_ok = all(
        (
            isinstance(np_img_, np.ndarray),
            np_img_.dtype.name == "float32",
            np_img_.shape == (181, 217, 181),
            np_img_[100, 100, 100] == 12175,
            np_img_[101, 100, 100] == 11972,
        )
    )
    assert to_numpy_ok


def test_to_sitk_image():
    """
    TEST_TO_SITK_IMAGE tests if the to_sitk_image function is implemented correctly.
    """
    img_ = ib.load_image("./data/T1native.nii.gz", False)
    np_img_ = ib.to_numpy_array(img_)
    rev_img_ = ib.to_sitk_image(np_img_, img_)
    to_sitk_ok = all(
        (
            isinstance(rev_img_, sitk.Image),
            rev_img_.GetOrigin() == img_.GetOrigin(),
            rev_img_.GetSpacing() == img_.GetSpacing(),
            rev_img_.GetDirection() == img_.GetDirection(),
            rev_img_.GetPixel(100, 100, 100) == 12175,
            rev_img_.GetPixel(100, 100, 101) == 11972,
        )
    )
    assert to_sitk_ok


def test_register():
    """
    TEST_REGISTER tests if the register_images function is implemented correctly.
    """
    img_ = ib.load_image("./data/T1native.nii.gz", False)
    atlas_img_ = ib.load_image("./data/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz", False)
    label_img_ = ib.load_image("./data/labels_native.nii.gz", True)
    if isinstance(atlas_img_, sitk.Image) and isinstance(label_img_, sitk.Image):
        registered_img_, registered_label_ = ib.register_images(
            img_, label_img_, atlas_img_
        )
        if isinstance(registered_img_, sitk.Image) and isinstance(
            registered_label_, sitk.Image
        ):
            stats = sitk.LabelStatisticsImageFilter()
            stats.Execute(registered_img_, registered_label_)
            labels = tuple(sorted(stats.GetLabels()))
            register_ok = all(
                (
                    registered_img_.GetSize()
                    == registered_label_.GetSize()
                    == (197, 233, 189),
                    labels == tuple(range(6)),
                )
            )
        else:
            register_ok = False
    else:
        register_ok = False
    assert register_ok


def test_preprocess_rescale_numpy():
    """
    TEST_PREPROCESS_RESCALE_NUMPY tests if the preprocess_rescale_numpy function
    is implemented correctly.
    """
    img_ = ib.load_image("./data/T1native.nii.gz", False)
    np_img_ = ib.to_numpy_array(img_)
    if isinstance(np_img_, np.ndarray):
        pre_np = ib.preprocess_rescale_numpy(np_img_, -3, 101)
        if isinstance(pre_np, np.ndarray):
            pre_np_ok = np.min(pre_np) == -3 and np.max(pre_np) == 101
        else:
            pre_np_ok = False
    else:
        pre_np_ok = False
    assert pre_np_ok


def test_preprocess_rescale_sitk():
    """
    TEST_PREPROCESS_RESCALE_SITK tests if the preprocess_rescale_sitk function
    is implemented correctly.
    """
    img_ = ib.load_image("./data/T1native.nii.gz", False)
    pre_sitk = ib.preprocess_rescale_sitk(img_, -3, 101)
    if isinstance(pre_sitk, sitk.Image):
        min_max = sitk.MinimumMaximumImageFilter()
        min_max.Execute(pre_sitk)
        pre_sitk_ok = min_max.GetMinimum() == -3 and min_max.GetMaximum() == 101
    else:
        pre_sitk_ok = False
    assert pre_sitk_ok


def test_extract_feature_median():
    """
    TEST_EXTRACT_FEATURE_MEDIAN tests if the extract_feature_median function
    is implemented correctly.
    """
    img_ = ib.load_image("./data/T1native.nii.gz", False)
    median_img_ = ib.extract_feature_median(img_)
    if isinstance(median_img_, sitk.Image):
        median_ref = ib.load_image("./data/T1med.nii.gz", False)
        if isinstance(median_ref, sitk.Image):
            min_max = sitk.MinimumMaximumImageFilter()
            min_max.Execute(median_img_ - median_ref)
            median_ok = min_max.GetMinimum() == 0 and min_max.GetMaximum() == 0
        else:
            median_ok = False
    else:
        median_ok = False
    assert median_ok


def test_postprocess_largest_component():
    """
    TEST_POSTPROCESS_LARGEST_COMPONENT tests if the postprocess_largest_component function
    is implemented correctly.
    """
    label_img_ = ib.load_image("./data/labels_native.nii.gz", True)
    largest_hippocampus = ib.postprocess_largest_component(
        label_img_ == 3
    )  # 3: hippocampus
    if isinstance(largest_hippocampus, sitk.Image):
        largest_ref = ib.load_image("./data/hippocampus_largest.nii.gz", True)
        if isinstance(largest_ref, sitk.Image):
            min_max = sitk.MinimumMaximumImageFilter()
            min_max.Execute(largest_hippocampus - largest_ref)
            post_ok = min_max.GetMinimum() == 0 and min_max.GetMaximum() == 0
        else:
            post_ok = False
    else:
        post_ok = False
    assert post_ok

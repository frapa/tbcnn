import numpy as np
import SimpleITK as sitk

def elastically_deform_image_2d(image, num_control_points, std_dev, interpolator='linear'):
    """Implements elastic deformations that are used for augmentation, using the pre-made SimpleITK library.
    The number of control points is (size of the grid - 2) due to borders, whilte std_dev is the standard
    deviation of the displacement vectors in pixels. Interpolator is either 'linear' or 'cubic'"""

    # Convert numpy array to itk object
    sitk_image = sitk.GetImageFromArray(image, isVector=False)

    # Allocate memory for transform parameters
    transform_mesh_size = [num_control_points] * sitk_image.GetDimension()
    transform = sitk.BSplineTransformInitializer(
        sitk_image ,
        transform_mesh_size
    )

    # Read the parameters as a numpy array, then add random
    # displacement and set the parameters back into the transform
    params = np.asarray(transform.GetParameters(), dtype=np.float64)
    params = params + np.random.randn(params.shape[0]) * std_dev
    transform.SetParameters(tuple(params))

    # Create resampler object
    # The interpolator can be set to sitk.sitkBSpline for cubic interpolation,
    # see https://itk.org/SimpleITKDoxygen/html/namespaceitk_1_1simple.html#a7cb1ef8bd02c669c02ea2f9f5aa374e5 for more options
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_image)
    resampler.SetInterpolator(sitk.sitkLinear if interpolator == 'linear' else sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(float(image.min())) # Fill with minimu value if out of image, float bcause otherwise tf thinks it's a double
    resampler.SetTransform(transform)

    # Execute augmentation
    sitk_augmented_image = resampler.Execute(sitk_image)

    # Convert back to numpy
    augmented_image = sitk.GetArrayFromImage(sitk_augmented_image)
    augmented_image = augmented_image.astype(dtype=np.float32)

    return augmented_image
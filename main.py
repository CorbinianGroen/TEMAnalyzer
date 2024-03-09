import numpy as np
import cv2
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.filters import threshold_sauvola
from skimage import measure, morphology, segmentation, feature, filters, util
from skimage.segmentation import find_boundaries
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import pandas as pd

# User-configurable parameters
low_cutoff = 30
high_cutoff = 400
a = 1.5
b = 5
window_size = 1351  # Size of the local region (must be odd)
k = 0.1  # Sensitivity parameter
border_width = 50  # Width of the border to mask out
scale_factor = 0.169  # nm per pixel for converting measurements
min_size = 150  # Minimum size threshold for particles
min_roundness = 0.75  # Minimum roundness threshold for particles
threshold_value = 0.3 #Trehshold for the distance map to exclude noise and get the larger peaks

file = '\\\\10.162.95.1\\data\\RRDE\\Corbi\\01_TEM\\20240126_CG15\\20240126_CG15_SA-MAG_X100k_011.jpg'  # Path to the input image

if __name__ == '__main__':
    # 1. Read the image
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

    # 2. Apply Fourier transform
    f_image = fft2(image)
    fshift = fftshift(f_image)

    # 3. Create a bandpass filter
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2

    # Cut-off frequencies
    low_cutoff = low_cutoff
    high_cutoff = high_cutoff

    # Create a mask first with 1s for a low-pass filter
    mask = np.zeros((rows, cols), dtype=np.float32)
    y, x = np.ogrid[:rows, :cols]
    center = (crow, ccol)
    dist_from_center = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)

    # Low pass filter mask
    mask[dist_from_center <= high_cutoff] = 1
    # High pass filter mask (inverted to create band-pass)
    mask[dist_from_center < low_cutoff] = 0

    # 4. Apply the filter
    fshift_filtered = fshift * mask

    # 5. Inverse Fourier Transform
    f_ishift = ifftshift(fshift_filtered)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    #saturate picture

    img_back = cv2.convertScaleAbs(img_back, alpha=a, beta=b)

    #convert to 8-bit (grayscale)
    img_back_8bit = np.uint8(cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX))

    img_uint8 = (img_back * 255).astype(np.uint8)

    # Apply Sauvola thresholding -> make a binary picture from the greyscale picture with local thresholding
    thresh_sauvola = threshold_sauvola(img_uint8, window_size=window_size, k=k)
    binary_sauvola = img_uint8 > thresh_sauvola

    # invert the picture to make particles white and background black -> needed for detection

    binary_sauvola_inverted = np.invert(binary_sauvola)


    # Compute the distance transform
    distance = ndimage.distance_transform_edt(binary_sauvola_inverted)

    # Apply a threshold to focus on significant regions, as previously done
    threshold_value = threshold_value * np.max(distance)
    thresholded_distance = np.where(distance > threshold_value, distance, 0)

    # Find peaks directly on the thresholded distance transform
    coordinates = feature.peak_local_max(thresholded_distance, min_distance=3, exclude_border=False)

    # Directly create unique markers for each peak without dilation
    markers = np.zeros_like(distance, dtype=int)
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)

    # Use the negative distance transform as input for watershed segmentation
    labels = segmentation.watershed(-thresholded_distance, markers, mask=binary_sauvola_inverted)

    #remove noise and select the particles
    props = measure.regionprops(labels)

    # Initialize a new label matrix for filtered regions
    filtered_labels = np.zeros_like(labels)

    # Assign new labels only to regions that meet the criteria
    new_label = 1
    for prop in props:
        roundness = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter else 0
        if prop.area > min_size and roundness > min_roundness:
            filtered_labels[labels == prop.label] = new_label
            new_label += 1


    def mask_edges(labels, border_width):
        masked_labels = labels.copy()
        masked_labels[:border_width, :] = 0  # Top edge
        masked_labels[-border_width:, :] = 0  # Bottom edge
        masked_labels[:, :border_width] = 0  # Left edge
        masked_labels[:, -border_width:] = 0  # Right edge
        return masked_labels


    # Example usage:
    filtered_labels = mask_edges(filtered_labels, border_width=border_width)  # Adjust `border_width` as needed

    # Find the boundaries of the filtered labels
    boundaries = segmentation.find_boundaries(filtered_labels)



    # Calculate properties of each region in the filtered_labels
    props_filtered = measure.regionprops(filtered_labels)

    # Initialize lists to store area and diameter values
    areas_nm2 = []
    diameters_nm = []


    for i, prop in enumerate(props_filtered, start=1):
        # Area in pixels
        area_pixels = prop.area
        # Convert area from pixels to nm^2
        area_nm2 = area_pixels * (scale_factor ** 2)

        # Diameter in pixels for a circle with the same area as the particle
        diameter_pixels = np.sqrt(4 * area_pixels / np.pi)
        # Convert diameter from pixels to nm
        diameter_nm = diameter_pixels * scale_factor

        # Store the calculated values
        areas_nm2.append(area_nm2)
        diameters_nm.append(diameter_nm)

    df_particles = pd.DataFrame({
        'Index': np.arange(1, len(areas_nm2) + 1),
        'Area_nm2': areas_nm2,
        'Diameter_nm': diameters_nm
    })

    # Calculate diameter squared and cubed, add as new columns
    df_particles['Diameter_nm_squared'] = df_particles['Diameter_nm'] ** 2
    df_particles['Diameter_nm_cubed'] = df_particles['Diameter_nm'] ** 3

    # Sum up the diameter squared and cubed columns
    sum_diameter_squared = df_particles['Diameter_nm_squared'].sum()
    sum_diameter_cubed = df_particles['Diameter_nm_cubed'].sum()

    # Calculate the division of the sum of cubed by the sum of squared diameters
    result = sum_diameter_cubed / sum_diameter_squared

    # Calculate the histogram
    bin_edges = np.arange(0, max(diameters_nm) + 0.5, 0.5)
    hist, bins = np.histogram(diameters_nm, bins=bin_edges, density=True)

    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_width = np.diff(bins)[0]  # Assuming uniform bin width

    # Extend the x values for fitting by one bin width at the start and end
    x_start = bins[0] - bin_width
    x_end = bins[-1] + bin_width
    x_extended = np.linspace(x_start, x_end, 100)

    # Fit a log-normal distribution to the diameters_nm data
    sigma, loc, scale = lognorm.fit(diameters_nm, floc=0)
    dist_extended = lognorm(s=sigma, loc=loc, scale=scale)

    # Generate PDF values for the extended x range
    pdf_fitted_extended = dist_extended.pdf(x_extended)

    # Find the maximum position (mode) of the log-normal distribution
    max_position = scale  # For a log-normal distribution

    # Calculate FWHM
    fwhm = np.exp(np.log(scale) + (sigma ** 2) / 2) * (np.exp(sigma ** 2) - 1) ** 0.5

    print(f"Max Position: {max_position} nm")
    print(f"FWHM: {fwhm} nm")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, hist, width=np.diff(bins), alpha=0.5, label='Normalized Histogram')
    plt.plot(x_extended, pdf_fitted_extended, label='Log-normal Fit Extended')
    plt.title('Nanoparticle Size Distribution')
    plt.xlabel('Diameter (nm)')
    plt.ylabel('Density')
    plt.legend()
    #plt.show()
    #print(df_particles, result)

    # Visualize the result

    # Create a 2x4 grid of subplots with shared x and y axes
    fig, ax = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)

    # Assume the necessary images are defined: binary_sauvola_inverted, distance, peaks_mask, labels, binary_sauvola, label_objects, filtered_image

    # First row of 4 images
    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 0].set_title('Binary Image')

    ax[0, 1].imshow(distance, cmap='plasma')
    ax[0, 1].set_title('Distance Transform')

    ax[0, 2].imshow(thresholded_distance, cmap='gray')
    ax[0, 2].set_title('Peaks Mask')

    ax[0, 3].imshow(labels, cmap='nipy_spectral')
    ax[0, 3].set_title('Watershed Segmentation')

    # Second row of 3 images (leave the last subplot empty)
    ax[1, 0].imshow(binary_sauvola, cmap='gray')
    ax[1, 0].set_title('Original Thresholded Image')

    ax[1, 1].imshow(labels, cmap='plasma')
    ax[1, 1].set_title('Labeled Image')

    ax[1, 2].imshow(boundaries, cmap='gray')
    ax[1, 2].set_title('Filtered by Size and Roundness')

    ax[1, 3].imshow(boundaries, cmap='gray')
    ax[1, 3].set_title('Filtered by Size and Roundness')

    for i, axi in enumerate(ax.flat[:-1]):

        axi.axis('off')  # Turn off axis for images


    plt.tight_layout()
    plt.show()

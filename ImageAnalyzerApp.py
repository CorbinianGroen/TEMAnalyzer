import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.filters import threshold_sauvola
from skimage import measure, morphology, segmentation, feature, filters, util
from skimage.segmentation import find_boundaries
from scipy.stats import lognorm


# Set the theme and color scheme
ctk.set_appearance_mode("dark")  # 'light' (default), 'dark'
ctk.set_default_color_theme("blue")  # 'blue' (default), 'green', 'dark-blue'

class ImageAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Analyzer")
        self.state('zoomed')  # Start the app maximized/full-screen

        # Set default parameters
        self.parameters = {
            'low_cutoff': 30,
            'high_cutoff': 400,
            'a': 1.5,
            'b': 5,
            'window_size': 1351,
            'k': 0.1,
            'border_width': 50,
            'min_size': 150,
            'min_roundness': 0.75,
            'treshold_value': 0.3,
            'min_distance': 3
        }
        self.current_image = None
        self.labels = None
        # Create the main layout frames
        self.frame_left = ctk.CTkFrame(self, width=0.7*self.winfo_screenwidth())
        self.frame_right = ctk.CTkFrame(self, width=0.3*self.winfo_screenwidth())

        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Set up the Matplotlib figure and canvas on the left side
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.frame_left)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Add the navigation toolbar to the canvas
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.frame_left)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Open image button
        self.open_button = ctk.CTkButton(self.frame_right, text="Open Image", command=self.open_image)
        self.open_button.pack(pady=10)

        # Parameters frame within the right frame
        self.parameters_frame = ctk.CTkFrame(self.frame_right)
        self.parameters_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Parameter: low_cutoff frequency
        self.low_cutoff_label = ctk.CTkLabel(self.parameters_frame, text="Low Cutoff Frequency")
        self.low_cutoff_label.pack()
        self.low_cutoff_entry = ctk.CTkEntry(self.parameters_frame)
        self.low_cutoff_entry.pack()
        self.low_cutoff_entry.insert(0, self.parameters['low_cutoff'])


        # Parameter: high_cutoff frequency
        self.high_cutoff_label = ctk.CTkLabel(self.parameters_frame, text="High Cutoff Frequency")
        self.high_cutoff_label.pack()
        self.high_cutoff_entry = ctk.CTkEntry(self.parameters_frame)
        self.high_cutoff_entry.pack()
        self.high_cutoff_entry.insert(0, self.parameters['high_cutoff'])
        self.high_cutoff_button = ctk.CTkButton(self.parameters_frame, text="Show", command=self.bandpass)
        self.high_cutoff_button.pack(pady=10)

        # Parameter: a
        self.a_label = ctk.CTkLabel(self.parameters_frame, text="a")
        self.a_label.pack()
        self.a_entry = ctk.CTkEntry(self.parameters_frame)
        self.a_entry.pack()
        self.a_entry.insert(0, self.parameters['a'])

        # Parameter: b
        self.b_label = ctk.CTkLabel(self.parameters_frame, text="b")
        self.b_label.pack()
        self.b_entry = ctk.CTkEntry(self.parameters_frame)
        self.b_entry.pack()
        self.b_entry.insert(0, self.parameters['b'])
        self.b_button = ctk.CTkButton(self.parameters_frame, text="Show", command=self.saturate)
        self.b_button.pack(pady=10)

        # Parameter: window size
        self.window_size_label = ctk.CTkLabel(self.parameters_frame, text="window size")
        self.window_size_label.pack()
        self.window_size_entry = ctk.CTkEntry(self.parameters_frame)
        self.window_size_entry.pack()
        self.window_size_entry.insert(0, self.parameters['window_size'])

        # Parameter: k
        self.k_label = ctk.CTkLabel(self.parameters_frame, text="k")
        self.k_label.pack()
        self.k_entry = ctk.CTkEntry(self.parameters_frame)
        self.k_entry.pack()
        self.k_entry.insert(0, self.parameters['k'])
        self.k_button = ctk.CTkButton(self.parameters_frame, text="Show", command=self.sauvola)
        self.k_button.pack(pady=10)

        # Parameter: border_width
        self.border_label = ctk.CTkLabel(self.parameters_frame, text="border width removal")
        self.border_label.pack()
        self.border_entry = ctk.CTkEntry(self.parameters_frame)
        self.border_entry.pack()
        self.border_entry.insert(0, self.parameters['border_width'])
        self.border_button = ctk.CTkButton(self.parameters_frame, text="Show", command=self.mask)
        self.border_button.pack(pady=10)

        # Parameter: treshold_value
        self.treshold_value_label = ctk.CTkLabel(self.parameters_frame, text="Particle detection")
        self.treshold_value_label.pack()
        self.treshold_value_entry = ctk.CTkEntry(self.parameters_frame)
        self.treshold_value_entry.pack()
        self.treshold_value_entry.insert(0, self.parameters['treshold_value'])
        self.treshold_value_button = ctk.CTkButton(self.parameters_frame, text="Show", command=self.distance)
        self.treshold_value_button.pack(pady=10)

        # Parameter: min_distance
        self.min_distance_label = ctk.CTkLabel(self.parameters_frame, text="Watershed")
        self.min_distance_label.pack()
        self.min_distance_entry = ctk.CTkEntry(self.parameters_frame)
        self.min_distance_entry.pack()
        self.min_distance_entry.insert(0, self.parameters['min_distance'])
        self.min_distance_button = ctk.CTkButton(self.parameters_frame, text="Show", command=self.watershed)
        self.min_distance_button.pack(pady=10)

        # Parameter: min_size
        self.min_size_label = ctk.CTkLabel(self.parameters_frame, text="min size")
        self.min_size_label.pack()
        self.min_size_entry = ctk.CTkEntry(self.parameters_frame)
        self.min_size_entry.pack()
        self.min_size_entry.insert(0, self.parameters['min_size'])

        # Parameter: min_roundness
        self.min_roundness_label = ctk.CTkLabel(self.parameters_frame, text="roundness")
        self.min_roundness_label.pack()
        self.min_roundness_entry = ctk.CTkEntry(self.parameters_frame)
        self.min_roundness_entry.pack()
        self.min_roundness_entry.insert(0, self.parameters['min_roundness'])
        self.min_roundness_button = ctk.CTkButton(self.parameters_frame, text="Show", command=self.particle)
        self.min_roundness_button.pack(pady=10)


        #OverlayButton
        self.overlay_button = ctk.CTkButton(self.parameters_frame, text="Overlay", command=self.overlay_current_image)
        self.overlay_button.pack(pady=10)

        # Add other parameter controls here similarly

        # Process button
        self.process_button = ctk.CTkButton(self.frame_right, text="Process Image", command=self.process_image)
        self.process_button.pack(pady=10)

        # Add button
        self.add_button = ctk.CTkButton(self.frame_right, text="Add Data", command=self.add_data)
        self.add_button.pack(pady=10)

        # Histogram button
        self.histogram_button = ctk.CTkButton(self.frame_right, text="Show Histogram", command=self.show_histogram)
        self.histogram_button.pack(pady=10)

        # Save button
        self.save_button = ctk.CTkButton(self.frame_right, text="Save Results", command=self.save_results)
        self.save_button.pack(pady=10)

        # Initialize dataframe to store results
        self.results_df = pd.DataFrame()

    # Define the functionality for each button and input here...
    def open_image(self):
        # Function to open and display a grayscale image with OpenCV using Matplotlib in Tkinter
        filepath = filedialog.askopenfilename()
        if not filepath:  # If no file is selected
            return

        # Read the image in grayscale
        self.image_cv = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        self.current_image = {'image_data': self.image_cv, 'process_type': 'open_image'}

        # Clear the existing figure
        self.figure.clf()

        # Create a new axis for displaying the image
        ax = self.figure.add_subplot(111)

        # Display the image on the new axis
        ax.imshow(self.image_cv, cmap='gray')
        ax.axis('off')  # Hide axis ticks and labels

        # Update the canvas to show the new image
        self.canvas.draw()


    def bandpass(self):
        f_image = fft2(self.image_cv)
        fshift = fftshift(f_image)

        # 3. Create a bandpass filter
        rows, cols = self.image_cv.shape
        crow, ccol = rows // 2, cols // 2

        # Cut-off frequencies
        low_cutoff = int(self.low_cutoff_entry.get())
        high_cutoff = int(self.high_cutoff_entry.get())

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
        self.img_back = np.abs(img_back)

        self.current_image = {'image_data': self.img_back, 'process_type': 'bandpass'}

        # Clear the existing figure
        self.figure.clf()

        # Create a new axis for displaying the image
        ax = self.figure.add_subplot(111)

        # Display the image on the new axis
        ax.imshow(self.img_back, cmap='gray')
        ax.axis('off')  # Hide axis ticks and labels

        # Update the canvas to show the new image
        self.canvas.draw()

    def saturate(self):

        self.img_back_sat = cv2.convertScaleAbs(self.img_back, alpha=float(self.a_entry.get()), beta=float(self.b_entry.get()))

        self.current_image = {'image_data': self.img_back_sat, 'process_type': 'saturate'}

        # Clear the existing figure
        self.figure.clf()

        # Create a new axis for displaying the image
        ax = self.figure.add_subplot(111)

        # Display the image on the new axis
        ax.imshow(self.img_back_sat, cmap='gray')
        ax.axis('off')  # Hide axis ticks and labels

        # Update the canvas to show the new image
        self.canvas.draw()

    def sauvola(self):

        # convert to 8-bit (grayscale)

        img_uint8 = (self.img_back_sat * 255).astype(np.uint8)

        # Apply Sauvola thresholding -> make a binary picture from the greyscale picture with local thresholding
        thresh_sauvola = threshold_sauvola(img_uint8, window_size=int(self.window_size_entry.get()), k=float(self.k_entry.get()))
        binary_sauvola = img_uint8 > thresh_sauvola

        # invert the picture to make particles white and background black -> needed for detection

        self.binary_sauvola_inverted = np.invert(binary_sauvola)

        self.current_image = {'image_data': self.binary_sauvola_inverted, 'process_type': 'sauvola'}

        # Clear the existing figure
        self.figure.clf()

        # Create a new axis for displaying the image
        ax = self.figure.add_subplot(111)

        # Display the image on the new axis
        ax.imshow(self.binary_sauvola_inverted, cmap='gray')
        ax.axis('off')  # Hide axis ticks and labels

        # Update the canvas to show the new image
        self.canvas.draw()

    def mask(self):
        border_width = int(self.border_entry.get())
        masked_sauvola = self.binary_sauvola_inverted
        masked_sauvola[:border_width, :] = 0  # Top edge
        masked_sauvola[-border_width:, :] = 0  # Bottom edge
        masked_sauvola[:, :border_width] = 0  # Left edge
        masked_sauvola[:, -border_width:] = 0

        self.masked_sauvola = masked_sauvola

        self.current_image = {'image_data': self.masked_sauvola, 'process_type': 'masked'}

        # Clear the existing figure
        self.figure.clf()

        # Create a new axis for displaying the image
        ax = self.figure.add_subplot(111)

        # Display the image on the new axis
        ax.imshow(self.masked_sauvola, cmap='gray')
        ax.axis('off')  # Hide axis ticks and labels

        # Update the canvas to show the new image
        self.canvas.draw()

    def distance(self):
        # Compute the distance transform
        self.distance = ndimage.distance_transform_edt(self.masked_sauvola)

        # Apply a threshold to focus on significant regions, as previously done
        threshold_value = float(self.treshold_value_entry.get()) * np.max(self.distance)
        self.thresholded_distance = np.where(self.distance > threshold_value, self.distance, 0)

        self.current_image = {'image_data': self.thresholded_distance, 'process_type': 'distance'}

        # Clear the existing figure
        self.figure.clf()

        # Create a new axis for displaying the image
        ax = self.figure.add_subplot(111)

        # Display the image on the new axis
        ax.imshow(self.thresholded_distance, cmap='gray')
        ax.axis('off')  # Hide axis ticks and labels

        # Update the canvas to show the new image
        self.canvas.draw()

    def watershed(self):

        coordinates = feature.peak_local_max(self.thresholded_distance, min_distance=int(self.min_distance_entry.get()), exclude_border=False)

        # Directly create unique markers for each peak without dilation
        markers = np.zeros_like(self.distance, dtype=int)
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)

        # Use the negative distance transform as input for watershed segmentation
        self.labels = segmentation.watershed(-self.thresholded_distance, markers, mask=self.masked_sauvola)

        self.current_image = {'image_data': self.labels, 'process_type': 'watershed'}

        # Clear the existing figure
        self.figure.clf()

        # Create a new axis for displaying the image
        ax = self.figure.add_subplot(111)

        # Display the image on the new axis
        ax.imshow(self.labels, cmap='nipy_spectral')
        ax.axis('off')  # Hide axis ticks and labels

        # Update the canvas to show the new image
        self.canvas.draw()

    def particle(self):

        props = measure.regionprops(self.labels)

        # Initialize a new label matrix for filtered regions
        self.filtered_labels = np.zeros_like(self.labels)

        # Assign new labels only to regions that meet the criteria
        new_label = 1
        for prop in props:
            roundness = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter else 0
            if prop.area > int(self.min_size_entry.get()) and roundness > float(self.min_roundness_entry.get()):
                self.filtered_labels[self.labels == prop.label] = new_label
                new_label += 1

        self.boundaries = segmentation.find_boundaries(self.filtered_labels)

        self.current_image = {'image_data': self.boundaries, 'process_type': 'particle'}

        # Clear the existing figure
        self.figure.clf()

        # Create a new axis for displaying the image
        ax = self.figure.add_subplot(111)

        # Display the image on the new axis
        ax.imshow(self.boundaries, cmap='gray')
        ax.axis('off')  # Hide axis ticks and labels

        # Update the canvas to show the new image
        self.canvas.draw()

    def overlay_current_image(self):
        if not hasattr(self, 'image_cv') or not hasattr(self, 'current_image'):
            print("Required attributes not set.")
            return

        process_type = self.current_image.get('process_type', '')
        image_data = self.current_image.get('image_data', None)

        # Immediately return for these process types
        if process_type in ['open_image', 'bandpass', 'saturate']:
            return

        original_bgr = cv2.cvtColor(self.image_cv, cv2.COLOR_GRAY2BGR) if len(
            self.image_cv.shape) == 2 else self.image_cv.copy()

        if process_type in ['sauvola', 'masked', 'particle']:
            # Convert image_data to uint8 if not already
            if image_data.dtype != np.uint8:
                if np.max(image_data) <= 1 and np.min(image_data) >= 0:  # Assuming image_data is normalized [0,1]
                    binary_image = (image_data * 255).astype(np.uint8)
                else:
                    binary_image = image_data.astype(np.uint8)
            else:
                binary_image = image_data
            # Create a binary mask to find non-black pixels
            mask = binary_image > 0
            overlay_image = original_bgr.copy()
            overlay_image[mask] = [0, 0, 255]  # Overlay in red where mask is true

        else:
            pass

        if process_type == 'watershed':
            # Apply nipy_spectral colormap to the watershed labels
            # First, normalize labels to 0-1 range for colormap application
            labels_normalized = image_data / np.max(image_data) if np.max(image_data) > 0 else image_data
            colormap_image = plt.get_cmap('nipy_spectral')(labels_normalized)
            # Convert to BGR format
            colored_labels = (colormap_image[..., :3] * 255).astype(np.uint8)
            colored_labels_bgr = cv2.cvtColor(colored_labels, cv2.COLOR_RGB2BGR)

            # Create a mask where watershed labels are not the background (assumed label=0)
            mask = image_data > 0

            # Initialize overlay_image with the original image
            overlay_image = original_bgr.copy()

            # Overlay colored labels on the original image only where mask is True
            overlay_image[mask] = colored_labels_bgr[mask]

        else:
            pass

        if process_type == 'distance':
            # Keep the distance processing as it works
            mask = image_data > 0
            colored_image = plt.get_cmap('plasma')(image_data / np.max(image_data)) if np.max(
                image_data) > 0 else plt.get_cmap('plasma')(image_data)
            colored_overlay = (colored_image[..., :3] * 255).astype(np.uint8)
            overlay_image = original_bgr.copy()
            overlay_image[mask] = colored_overlay[mask]

        else:
            pass
        '''
        elif process_type in ['distance', 'watershed']:
            # Keep the distance processing as it works
            mask = image_data > 0
            colormap = 'plasma' if process_type == 'distance' else 'nipy_spectral'
            colored_image = plt.get_cmap(colormap)(image_data / np.max(image_data)) if np.max(
                image_data) > 0 else plt.get_cmap(colormap)(image_data)
            colored_overlay = (colored_image[..., :3] * 255).astype(np.uint8)
            overlay_image = original_bgr.copy()
            # Apply the colored overlay where mask is True
            overlay_image[mask] = colored_overlay[mask]
        
        elif:
            print(f"Unrecognized process type: {process_type}")
            return'''

        # Update the Matplotlib figure for display
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        self.canvas.draw()

    def process_image(self):
        # Process the image with current parameters
        pass  # Implement image processing logic

    def add_data(self):
        # Add processed data to the dataframe
        pass  # Implement data addition logic

    def show_histogram(self):
        # Show the histogram of the collected data
        pass  # Implement histogram display logic

    def save_results(self):
        # Save the results and parameters
        pass  # Implement results saving logic

# Run the app
if __name__ == '__main__':
    app = ImageAnalyzerApp()
    app.mainloop()



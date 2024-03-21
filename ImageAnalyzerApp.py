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
from skimage import measure, segmentation, feature, morphology
from scipy.stats import lognorm
from concurrent.futures import ProcessPoolExecutor
import os
import json

# add a range of what number should be
# benchmark it -> compare lose boundaries to thight boundaries and deselect the errenous ones
# compared to handpicked

class ToolTip(object):
    def __init__(self, widget, text='widget info'):
        self.widget = widget
        self.text = text
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)
        self.tooltip_window = None

    def enter(self, event=None):
        if self.tooltip_window:
            return

        # Get widget bounding box
        x, y, cx, cy = self.widget.bbox("insert")
        # Calculate position relative to root window
        x = self.widget.winfo_rootx() - 160
        y += self.widget.winfo_rooty() + 20

        # Create the tooltip window
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)

        # Check if tooltip goes beyond the screen width
        screen_width = self.widget.winfo_screenwidth()
        tooltip_width = 150  # Assuming a fixed tooltip width for simplicity, adjust as needed
        if x + tooltip_width > screen_width:
            x = self.widget.winfo_rootx() - tooltip_width - 10  # Position to the left of the widget

        self.tooltip_window.wm_geometry("+%d+%d" % (x, y))

        label = tk.Label(self.tooltip_window, text=self.text, justify='left',
                         background="#f0f0f0", relief='flat', borderwidth=0,
                         font=("times", "8", "normal"))
        label.pack(ipadx=1)

    def leave(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None



# Set the theme and color scheme
ctk.set_appearance_mode("dark")  # 'light' (default), 'dark'
ctk.set_default_color_theme("blue")  # 'blue' (default), 'green', 'dark-blue'


def sauvola_processing(params):
    img, window_size, k = params['img'], params['window_size'], params['k']
    img_uint8 = (img * 255).astype(np.uint8)
    if window_size % 2 == 0:
        window_size = window_size + 1


    thresh_sauvola = threshold_sauvola(img_uint8, window_size=window_size, k=k)
    binary_sauvola = img_uint8 > thresh_sauvola
    binary_sauvola_inverted = np.invert(binary_sauvola)
    return binary_sauvola_inverted


def particle_processing(labels, min_size, min_roundness):
    """
    Standalone function to perform particle analysis in parallel.
    This function does not interact with the GUI directly.
    """
    props = measure.regionprops(labels)
    filtered_labels = np.zeros_like(labels)
    new_label = 1
    for prop in props:
        roundness = (4 * np.pi * prop.area) / (prop.perimeter ** 2) if prop.perimeter else 0
        if prop.area >= min_size and roundness >= min_roundness:
            filtered_labels[labels == prop.label] = new_label
            new_label += 1

    boundaries = segmentation.find_boundaries(filtered_labels)
    return filtered_labels, boundaries


def watershed_processing(params):
    from skimage import feature, segmentation
    import numpy as np

    thresholded_distance, distances, masked_sauvola = params['thresholded_distance'], params['distances'], params[
        'masked_sauvola']
    min_distance = params['min_distance']

    coordinates = feature.peak_local_max(thresholded_distance, min_distance=min_distance, exclude_border=False)
    markers = np.zeros_like(distances, dtype=int)
    markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
    labels = segmentation.watershed(-thresholded_distance, markers, mask=masked_sauvola)

    return labels


class ImageAnalyzerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Image Analyzer")
        self.state('zoomed')  # Start the app maximized/full-screen

        self.executor = ProcessPoolExecutor(max_workers=4)
        self.processing_completed = False

        self.filenames = []

        # Set default parameters
        self.parameters = {
            'low_cutoff': 30,
            'high_cutoff': 400,
            'blur': 251,
            'a': 1.5,
            'b': 5,
            'window_size': 1351,
            'k': 0.1,
            'border_width': 50,
            'min_size': 150,
            'min_roundness': 0.75,
            'threshold_value': 0.3,
            'min_distance': 3,
            'scale_factor': 0.169,
            'density': 21.5
        }

        self.load_parameters()

        self.current_image = None
        self.labels = None
        # Create the main layout frames
        self.frame_left = ctk.CTkFrame(self)
        self.frame_right = ctk.CTkFrame(self)

        # Assuming you want the right frame to take minimal necessary space
        # and the left frame (canvas frame) to take up the remaining space.
        self.frame_right.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

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
        self.open_button.pack(pady=5)

        # Initialize Progress Bar in the right frame below the open button
        self.progress_bar = ctk.CTkProgressBar(self.frame_right, width=200, height=20)
        self.progress_bar.pack(pady=5)
        self.progress_bar.set(0)  # Set progress to 0%

        # Parameters frame within the right frame
        self.parameters_frame = ctk.CTkFrame(self.frame_right)
        self.parameters_frame.pack(fill=tk.BOTH, pady=10)

        # Use grid layout management within self.parameters_frame
        self.parameters_frame.grid_columnconfigure(0, weight=1)
        self.parameters_frame.grid_columnconfigure(1, weight=1)
        self.parameters_frame.grid_columnconfigure(2, weight=1)

        # Parameter: low_cutoff frequency
        self.low_cutoff_label = ctk.CTkLabel(self.parameters_frame, text="Low Cutoff Frequency")
        # self.low_cutoff_label.pack()
        self.low_cutoff_entry = ctk.CTkEntry(self.parameters_frame)
        # self.low_cutoff_entry.pack()
        self.low_cutoff_entry.insert(0, self.parameters['low_cutoff'])

        # Parameter: high_cutoff frequency
        self.high_cutoff_label = ctk.CTkLabel(self.parameters_frame, text="High Cutoff Frequency")
        # self.high_cutoff_label.pack()
        self.high_cutoff_entry = ctk.CTkEntry(self.parameters_frame)
        # self.high_cutoff_entry.pack()
        self.high_cutoff_entry.insert(0, self.parameters['high_cutoff'])
        self.high_cutoff_button = ctk.CTkButton(self.parameters_frame, text="Show", command=lambda: self.set_and_start_processing('bandpass', self.bandpass))
        # self.high_cutoff_button.pack(pady=10)

        self.low_cutoff_label.grid(row=0, column=0, pady=(5, 0), padx=(5, 2))
        self.low_cutoff_entry.grid(row=1, column=0, pady=(2, 5), padx=(5, 2))
        self.high_cutoff_label.grid(row=0, column=1, pady=(5, 0), padx=(2, 5))
        self.high_cutoff_entry.grid(row=1, column=1, pady=(2, 5), padx=(2, 5))
        self.high_cutoff_button.grid(row=1, column=2, pady=(2, 5), padx=(2, 5), sticky="ew")

        ToolTip(widget=self.low_cutoff_label, text='High Cutoff Frequency: This is the upper threshold frequency for the bandpass filter.\nFrequencies higher than this value will be attenuated,\nhelping to remove high-frequency noise from the image.')
        ToolTip(widget=self.high_cutoff_label, text='Low Cutoff Frequency: This is the lower threshold frequency for the bandpass filter.\nFrequencies lower than this value will be attenuated,\nwhich helps in removing background variation and large-scale structures that are not of interest.')

        # Add a switch to toggle between frequency control visibility
        self.frequency_switch = ctk.CTkSwitch(self.parameters_frame, text="Toggle Frequency",
                                              command=self.toggle_frequency_controls)
        self.frequency_switch.grid(row=0, column=2, pady=(2, 5), padx=(2, 5))

        # Initialize new controls for alternative processing option but don't place them in the grid yet
        self.blur_label = ctk.CTkLabel(self.parameters_frame, text="Gaussian Blur")
        self.blur_entry = ctk.CTkEntry(self.parameters_frame)
        self.blur_entry.insert(0, self.parameters['blur'])
        self.blur_button = ctk.CTkButton(self.parameters_frame, text="Show", command=lambda: self.set_and_start_processing('bandpass', self.blur))

        # You can initially hide the new option controls
        self.blur_label.grid_remove()
        self.blur_entry.grid_remove()
        self.blur_button.grid_remove()



        # Parameter: a
        self.a_label = ctk.CTkLabel(self.parameters_frame, text="a")
        # self.a_label.pack()
        self.a_entry = ctk.CTkEntry(self.parameters_frame)
        # self.a_entry.pack()
        self.a_entry.insert(0, self.parameters['a'])

        # Parameter: b
        self.b_label = ctk.CTkLabel(self.parameters_frame, text="b")
        # self.b_label.pack()
        self.b_entry = ctk.CTkEntry(self.parameters_frame)
        # self.b_entry.pack()
        self.b_entry.insert(0, self.parameters['b'])
        self.b_button = ctk.CTkButton(self.parameters_frame, text="Show", command=lambda: self.set_and_start_processing('saturate', self.saturate))
        # self.b_button.pack(pady=10)

        self.a_label.grid(row=2, column=0, pady=(5, 2), padx=(5, 2))
        self.b_label.grid(row=2, column=1, pady=(5, 2), padx=(2, 5))
        self.a_entry.grid(row=3, column=0, pady=(2, 5), padx=(5, 2))
        self.b_entry.grid(row=3, column=1, pady=(2, 5), padx=(2, 5))
        self.b_button.grid(row=3, column=2, pady=(2, 5), padx=(2, 5), sticky="ew")

        ToolTip(widget=self.a_label, text='This parameter controls the contrast of the image during saturation.\nIncreasing alpha enhances the contrast, making the particles\nmore distinguishable from the background.')
        ToolTip(widget=self.b_label, text='This parameter adjusts the brightness of the image post-contrast adjustment.\nIncreasing beta lightens the image, while decreasing it makes the image darker.')

        # Parameter: window size
        self.window_size_label = ctk.CTkLabel(self.parameters_frame, text="window size")
        # self.window_size_label.pack()
        self.window_size_entry = ctk.CTkEntry(self.parameters_frame)
        # self.window_size_entry.pack()
        self.window_size_entry.insert(0, self.parameters['window_size'])

        # Parameter: k
        self.k_label = ctk.CTkLabel(self.parameters_frame, text="k")
        # self.k_label.pack()
        self.k_entry = ctk.CTkEntry(self.parameters_frame)
        # self.k_entry.pack()
        self.k_entry.insert(0, self.parameters['k'])
        self.k_button = ctk.CTkButton(self.parameters_frame, text="Show", command=lambda: self.set_and_start_processing('sauvola', self.sauvola))
        # self.k_button.pack(pady=10)

        self.window_size_label.grid(row=4, column=0, pady=(5, 0), padx=(5, 2))
        self.k_label.grid(row=4, column=1, pady=(5, 0), padx=(2, 5))
        self.window_size_entry.grid(row=5, column=0, pady=(2, 5), padx=(5, 2))
        self.k_entry.grid(row=5, column=1, pady=(2, 5), padx=(2, 5))
        self.k_button.grid(row=5, column=2, pady=(2, 5), padx=(2, 5), sticky="ew")

        ToolTip(widget=self.window_size_label, text='Window Size: The size of the local region around each pixel\nfor which the threshold is calculated in Sauvola thresholding.\nA larger window considers more of the local context,\nwhich is useful for varying backgrounds.')
        ToolTip(widget=self.k_label, text='k: A parameter that controls the sensitivity of the Sauvola method to local variations in contrast.\nHigher values make the algorithm more sensitive to shadows and other subtle features..')

        # Parameter: border_width
        self.border_label = ctk.CTkLabel(self.parameters_frame, text="border width removal")
        # self.border_label.pack()
        self.border_entry = ctk.CTkEntry(self.parameters_frame)
        # self.border_entry.pack()
        self.border_entry.insert(0, self.parameters['border_width'])
        self.border_button = ctk.CTkButton(self.parameters_frame, text="Show", command=lambda: self.set_and_start_processing('mask', self.mask))
        # self.border_button.pack(pady=10)

        self.border_label.grid(row=6, column=1, pady=(5, 0), padx=(5, 2))
        self.border_entry.grid(row=7, column=1, pady=(2, 5), padx=(5, 2))
        self.border_button.grid(row=7, column=2, pady=(2, 5), padx=(2, 5), sticky="ew")

        ToolTip(widget=self.border_label, text='Border Width: The width of the border removed from the edge of the image.\nThis is used to eliminate edge effects and artifacts that are not representative of the sample.')

        # Parameter: threshold_value
        self.threshold_value_label = ctk.CTkLabel(self.parameters_frame, text="Particle detection")
        # self.threshold_value_label.pack()
        self.threshold_value_entry = ctk.CTkEntry(self.parameters_frame)
        # self.threshold_value_entry.pack()
        self.threshold_value_entry.insert(0, self.parameters['threshold_value'])
        self.threshold_value_button = ctk.CTkButton(self.parameters_frame, text="Show", command=lambda: self.set_and_start_processing('distance', self.distance))
        # self.threshold_value_button.pack(pady=10)

        self.threshold_value_label.grid(row=8, column=1, pady=(5, 0), padx=(5, 2))
        self.threshold_value_entry.grid(row=9, column=1, pady=(2, 5), padx=(5, 2))
        self.threshold_value_button.grid(row=9, column=2, pady=(2, 5), padx=(2, 5), sticky="ew")

        ToolTip(widget=self.threshold_value_label, text='Threshold Value: In the context of the distance transform,\nthis threshold value is used to isolate prominent features from the background\nby focusing on regions that are sufficiently far from the background.')

        # Parameter: min_distance
        self.min_distance_label = ctk.CTkLabel(self.parameters_frame, text="Watershed")
        # self.min_distance_label.pack()
        self.min_distance_entry = ctk.CTkEntry(self.parameters_frame)
        # self.min_distance_entry.pack()
        self.min_distance_entry.insert(0, self.parameters['min_distance'])
        self.min_distance_button = ctk.CTkButton(self.parameters_frame, text="Show", command=lambda: self.set_and_start_processing('watershed', self.watershed))
        # self.min_distance_button.pack(pady=10)

        self.min_distance_label.grid(row=10, column=1, pady=(5, 0), padx=(5, 2))
        self.min_distance_entry.grid(row=11, column=1, pady=(2, 5), padx=(5, 2))
        self.min_distance_button.grid(row=11, column=2, pady=(2, 5), padx=(2, 5), sticky="ew")

        ToolTip(widget=self.min_distance_label, text='Min Distance: The minimum distance between peaks in the distance transform\nused in watershed segmentation. It helps in ensuring that the watershed algorithm\ndoes not over-segment the image by considering too closely spaced local maxima\nas separate particles.')

        # Parameter: min_size
        self.min_size_label = ctk.CTkLabel(self.parameters_frame, text="min size")
        # self.min_size_label.pack()
        self.min_size_entry = ctk.CTkEntry(self.parameters_frame)
        # self.min_size_entry.pack()
        self.min_size_entry.insert(0, self.parameters['min_size'])

        # Parameter: min_roundness
        self.min_roundness_label = ctk.CTkLabel(self.parameters_frame, text="roundness")
        # self.min_roundness_label.pack()
        self.min_roundness_entry = ctk.CTkEntry(self.parameters_frame)
        # self.min_roundness_entry.pack()
        self.min_roundness_entry.insert(0, self.parameters['min_roundness'])
        self.min_roundness_button = ctk.CTkButton(self.parameters_frame, text="Show", command=lambda: self.set_and_start_processing('particle', self.particle))
        # self.min_roundness_button.pack(pady=10)

        self.min_size_label.grid(row=12, column=0, pady=(5, 0), padx=(5, 2))
        self.min_roundness_label.grid(row=12, column=1, pady=(5, 0), padx=(2, 5))
        self.min_size_entry.grid(row=13, column=0, pady=(2, 5), padx=(5, 2))
        self.min_roundness_entry.grid(row=13, column=1, pady=(2, 5), padx=(2, 5))
        self.min_roundness_button.grid(row=13, column=2, pady=(2, 5), padx=(2, 5), sticky="ew")

        ToolTip(widget=self.min_size_label, text='Min Size: The minimum size (area in pixels)\na particle must have to be considered in the analysis.\nThis helps in excluding too small particles that might be\nnoise or irrelevant.')
        ToolTip(widget=self.min_roundness_label, text='Min Roundness: The minimum roundness a particle must have to be included in the analysis.\nRoundness is a measure of how circular a particle is,\nhelping to exclude particles that are too elongated or irregular.')

        # OverlayButton
        self.overlay_button = ctk.CTkButton(self.parameters_frame, text="Overlay", command=self.overlay_current_image)
        # self.overlay_button.pack(pady=10)
        self.overlay_button.grid(row=15, column=2, pady=(5, 10), padx=(5, 10), sticky="ew")

        # Parameter: scale_factor
        self.scale_factor_label = ctk.CTkLabel(self.parameters_frame, text="nm / pixel")
        # self.scale_factor_label.pack()
        self.scale_factor_entry = ctk.CTkEntry(self.parameters_frame)
        # self.scale_factor_entry.pack()
        self.scale_factor_entry.insert(0, self.parameters['scale_factor'])

        self.scale_factor_label.grid(row=16, column=1, pady=(5, 0), padx=(2, 5))
        self.scale_factor_entry.grid(row=17, column=1, pady=(2, 5), padx=(2, 5))

        ToolTip(widget=self.scale_factor_label, text='Scale Factor: The scale factor relates pixel size to physical distance,\nallowing measurements in the image to be converted to real-world units')

        # Parameter: density
        self.density_label = ctk.CTkLabel(self.parameters_frame, text="NM Density")
        # self.scale_factor_label.pack()
        self.density_entry = ctk.CTkEntry(self.parameters_frame)
        # self.scale_factor_entry.pack()
        self.density_entry.insert(0, self.parameters['density'])

        self.density_label.grid(row=16, column=0, pady=(5, 0), padx=(5, 2))
        self.density_entry.grid(row=17, column=0, pady=(2, 5), padx=(5, 2))

        ToolTip(widget=self.density_label, text='Density: Defines the density of the particles analyzed. This will be used for the calculation of the expected ECSA')

        # Deselection initialization
        self.deselect_mode_switch = ctk.CTkSwitch(self.parameters_frame, text="Deselection Mode", command=self.toggle_deselect_mode)
        self.deselect_mode_switch.grid(row=16, column=2, pady=(2, 5), padx=(2, 5))
        self.deselect_mode = False

        self.particles_df = pd.DataFrame(columns=['Label', 'Pixels'])

        # Selection initialization
        self.select_mode_switch = ctk.CTkSwitch(self.parameters_frame, text="Selection Mode",
                                                  command=self.toggle_select_mode)
        self.select_mode_switch.grid(row=17, column=2, pady=(2, 5), padx=(2, 5))
        self.select_mode = False

        self.particles_df_added = pd.DataFrame(columns=['Label', 'Pixels'])

        self.process_buttons_frame = ctk.CTkFrame(self.frame_right)
        self.process_buttons_frame.pack(fill=tk.BOTH, pady=0, expand=False)

        self.process_buttons_frame.grid_columnconfigure(0, weight=1)
        self.process_buttons_frame.grid_columnconfigure(1, weight=1)
        self.process_buttons_frame.grid_columnconfigure(2, weight=1)

        # Create and position the Save Configuration button
        self.save_config_button = ctk.CTkButton(self.process_buttons_frame, text="Save Configuration",
                                                command=self.save_parameters)
        self.save_config_button.grid(row=0, column=0, padx=5, pady=0,
                                     sticky='ew')  # Adjust padx and pady as needed for spacing

        # Create and position the Reset View button
        self.reset_view_button = ctk.CTkButton(self.process_buttons_frame, text="Reset View", command=self.reset_view)
        self.reset_view_button.grid(row=0, column=1, padx=5, pady=0,
                                    sticky='ew')  # Adjust padx and pady as needed for spacing

        # Create and position the Process Image button
        self.process_button = ctk.CTkButton(self.process_buttons_frame, text="Process Image", command=self.process_data)
        self.process_button.grid(row=0, column=2, padx=5, pady=0,
                                 sticky='ew')  # Adjust padx and pady as needed for spacing
        # Results display frame
        self.results_frame = ctk.CTkScrollableFrame(self.frame_right, width=300, height=100, corner_radius=0, fg_color="transparent")
        self.results_frame.pack(fill='both', expand=True)
        self.setup_results_display()

        self.control_buttons_frame = ctk.CTkFrame(self.frame_right)
        self.control_buttons_frame.pack(fill=tk.BOTH, pady=5, expand=False)

        # Add and Save buttons
        self.add_button = ctk.CTkButton(self.control_buttons_frame, text="Clear", command=self.clear_data)
        self.add_button.pack(side=tk.LEFT, padx=5, pady=10, fill=tk.X, expand=True)

        self.save_button = ctk.CTkButton(self.control_buttons_frame, text="Save Results", command=self.save_results)
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=10, fill=tk.X, expand=True)

    def toggle_frequency_controls(self):
        if self.frequency_switch.get():  # If the switch is on
            # Hide low and high frequency controls
            self.low_cutoff_label.grid_remove()
            self.low_cutoff_entry.grid_remove()
            self.high_cutoff_label.grid_remove()
            self.high_cutoff_entry.grid_remove()
            self.high_cutoff_button.grid_remove()

            # Show new controls for the alternative processing option
            self.blur_label.grid(row=0, column=0, pady=(5, 0), padx=(5, 2))
            self.blur_entry.grid(row=1, column=0, pady=(2, 5), padx=(5, 2))
            self.blur_button.grid(row=1, column=2, pady=(2, 5), padx=(2, 5), sticky="ew")
        else:
            # Show low and high frequency controls
            self.low_cutoff_label.grid()
            self.low_cutoff_entry.grid()
            self.high_cutoff_label.grid()
            self.high_cutoff_entry.grid()
            self.high_cutoff_button.grid()

            # Hide new controls for the alternative processing option
            self.blur_label.grid_remove()
            self.blur_entry.grid_remove()
            self.blur_button.grid_remove()

    def load_parameters(self):
        try:
            with open('config.json', 'r') as config_file:
                self.parameters = json.load(config_file)
        except (FileNotFoundError, json.JSONDecodeError):
            return

    def save_parameters(self):
        with open('config.json', 'w') as config_file:
            self.parameters = {
                'low_cutoff': self.low_cutoff_entry.get(),
                'high_cutoff': self.high_cutoff_entry.get(),
                'blur': self.blur_entry.get(),
                'a': self.a_entry.get(),
                'b': self.b_entry.get(),
                'window_size': self.window_size_entry.get(),
                'k': self.k_entry.get(),
                'border_width': self.border_entry.get(),
                'min_size': self.min_size_entry.get(),
                'min_roundness': self.min_roundness_entry.get(),
                'threshold_value': self.threshold_value_entry.get(),
                'min_distance': self.min_distance_entry.get(),
                'scale_factor': self.scale_factor_entry.get(),
                'density': self.density_entry.get()
            }


            json.dump(self.parameters, config_file, indent=4)

    def setup_results_display(self):
        # Labels for displaying results
        self.results_labels = {
            'filenames': ctk.CTkLabel(self.results_frame, text="Filenames:"),
            'number_of_particles': ctk.CTkLabel(self.results_frame, text="Number of Particles:"),
            'average_diameter': ctk.CTkLabel(self.results_frame, text="Average Diameter:"),
            'surface_average': ctk.CTkLabel(self.results_frame, text="Surface Average:"),
            'expected_ecsa': ctk.CTkLabel(self.results_frame, text="Expected ECSA:"),
            'max_position': ctk.CTkLabel(self.results_frame, text="Max Position:"),
            'fwhm': ctk.CTkLabel(self.results_frame, text="FWHM:")
        }

        # Positioning results labels
        for i, label in enumerate(self.results_labels.values()):
            label.grid(row=i, column=0, sticky='w', padx=5, pady=(0, 5))

    def update_results_display(self, results):
        self.results_labels['filenames'].configure(text=f"Filenames: {results['Filenames']}")
        self.results_labels['number_of_particles'].configure(text=f"Number of Particles: {results['Number_of_Paticles']}")
        self.results_labels['average_diameter'].configure(text=f"Average Diameter: {results['Average_Diameter_nm']:.2f} \u00B1 {results['Average_Diameter_nm_std']:.2f} nm")
        self.results_labels['surface_average'].configure(text=f"Surface Average: {results['Surface_Average']:.2f} nm")
        self.results_labels['expected_ecsa'].configure(text=f"Expected ECSA: {results['Expected_ECSA_m²/g']:.2f} m²/g")
        self.results_labels['max_position'].configure(text=f"Max Position: {results['Max_Position_nm']:.2f} nm")
        self.results_labels['fwhm'].configure(text=f"FWHM: {results['FWHM_nm']:.2f} nm")

    def toggle_deselect_mode(self):
        activate = self.deselect_mode_switch.get()  # This should return True or False based on the switch state

        if activate:
            self.cid = self.figure.canvas.mpl_connect('button_press_event', self.on_figure_click)
            self.deselect_mode = True
            # Disable the select mode switch
            self.select_mode_switch.configure(state="disabled")
        else:
            if self.cid is not None:
                self.figure.canvas.mpl_disconnect(self.cid)
                self.cid = None
            self.deselect_mode = False
            # Re-enable the select mode switch
            self.select_mode_switch.configure(state="normal")

    def toggle_select_mode(self):
        activate = self.select_mode_switch.get()  # This should return True or False based on the switch state

        if activate:
            self.cid = self.figure.canvas.mpl_connect('button_press_event', self.on_figure_click_select)
            self.select_mode = True
            # Disable the deselect mode switch
            self.deselect_mode_switch.configure(state="disabled")
        else:
            if self.cid is not None:
                self.figure.canvas.mpl_disconnect(self.cid)
                self.cid = None
            self.select_mode = False
            # Re-enable the deselect mode switch
            self.deselect_mode_switch.configure(state="normal")

    def set_and_start_processing(self, process_name, process_function):
        self.initial_process = process_name
        self.processing_completed = False
        process_function()
        self.start_processing_check()

    def start_processing_check(self):
        """Starts a loop that checks if processing is completed."""
        if self.processing_completed:
            self.plot_result_based_on_initial_process()
        else:
            # Recheck after a short delay
            self.after(100, self.start_processing_check)

    def plot_result_based_on_initial_process(self):
        """Plots the result based on the initial process."""
        if self.initial_process == "bandpass":
            self.plotting(image=self.img_back, cmap='gray')
            self.current_image = {'image_data': self.img_back, 'process_type': 'bandpass'}
        elif self.initial_process == "saturate":
            self.plotting(image=self.img_back_sat, cmap='gray')
            self.current_image = {'image_data': self.img_back_sat, 'process_type': 'saturate'}
        elif self.initial_process == "sauvola":
            self.plotting(image=self.binary_sauvola_inverted, cmap='gray')
            self.current_image = {'image_data': self.binary_sauvola_inverted, 'process_type': 'sauvola'}
        elif self.initial_process == "mask":
            self.plotting(image=self.masked_sauvola, cmap='gray')
            self.current_image = {'image_data': self.masked_sauvola, 'process_type': 'mask'}
        elif self.initial_process == "distance":
            self.plotting(image=self.thresholded_distance, cmap='gray')
            self.current_image = {'image_data': self.thresholded_distance, 'process_type': 'distance'}
        elif self.initial_process == "watershed":
            self.plotting(image=self.labels, cmap='nipy_spectral')
            self.current_image = {'image_data': self.labels, 'process_type': 'watershed'}
        elif self.initial_process == "particle":
            self.plotting(image=self.boundaries, cmap='gray')
            self.current_image = {'image_data': self.boundaries, 'process_type': 'particle'}
        # Reset the flag
        self.processing_completed = False

    def plotting(self, image, cmap=None, xlim=None, ylim=None):
        canvas_width, canvas_height = self.canvas_widget.winfo_width(), self.canvas_widget.winfo_height()
        dpi = self.figure.dpi
        fig_width = canvas_width / dpi
        fig_height = canvas_height / dpi

        # Adjust figure size based on the canvas size
        self.figure.set_size_inches(fig_width, fig_height, forward=True)
        self.figure.clf()
        ax = self.figure.add_subplot(111)

        # Preserve the aspect ratio of the image
        img_height, img_width = image.shape[:2]
        img_aspect = img_width / img_height
        canvas_aspect = canvas_width / canvas_height

        # Determine scaling to fit the image within the canvas
        if img_aspect > canvas_aspect:
            # Image is wider than canvas
            scale = canvas_width / img_width
        else:
            # Image is taller than canvas
            scale = canvas_height / img_height

        # Calculate new image dimensions
        new_img_width = img_width * scale / dpi
        new_img_height = img_height * scale / dpi

        # Center the image in the figure
        left_margin = (fig_width - new_img_width) / 2
        bottom_margin = (fig_height - new_img_height) / 2

        # Adjust subplot parameters to fit the image
        self.figure.subplots_adjust(left=left_margin / fig_width,
                                    right=1 - (left_margin / fig_width),
                                    bottom=bottom_margin / fig_height,
                                    top=1 - (bottom_margin / fig_height),
                                    wspace=0, hspace=0)

        if cmap is None:
            ax.imshow(image, aspect='equal')
        else:
            ax.imshow(image, cmap=cmap, aspect='equal')

        ax.axis('off')

        if xlim and ylim:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        self.canvas.draw()

    def open_image(self):
        self.processing_completed = False
        # Function to open and display a grayscale image with OpenCV using Matplotlib in Tkinter
        self.filepath = filedialog.askopenfilename()
        if not self.filepath:  # If no file is selected
            return
        self.progress_bar.set(0)
        self.update_idletasks()
        # Read the image in grayscale
        self.image_cv = cv2.imread(self.filepath, cv2.IMREAD_GRAYSCALE)

        self.original_image_size = self.image_cv.shape[::-1]  # This reverses the tuple to (width, height)

        self.current_image = {'image_data': self.image_cv, 'process_type': 'open_image'}

        self.plotting(image=self.image_cv, cmap='gray')

        self.process_all()

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

        self.saturate()  # Apply saturation

    def blur(self):

        ksize = int(self.blur_entry.get())
        if ksize % 2 == 0:
            ksize += 1  # Make ksize odd if it's even

        blurred = cv2.GaussianBlur(self.image_cv, (ksize, ksize), 0)

        self.img_back = cv2.subtract(blurred, self.image_cv)

        self.saturate()  # Apply saturation



    def saturate(self):

        self.img_back_sat = cv2.convertScaleAbs(self.img_back, alpha=float(self.a_entry.get()), beta=float(self.b_entry.get()))

        self.sauvola()  # Apply Sauvola thresholding

    def sauvola(self):

        params = {
            'img': self.img_back_sat,  # Example image data
            'window_size': int(self.window_size_entry.get()),
            'k': float(self.k_entry.get())
        }

        future = self.executor.submit(sauvola_processing, params)
        future.add_done_callback(lambda future: self.on_sauvola_completed(future, 'gray'))
        # future.add_done_callback(self.on_sauvola_completed)

    def on_sauvola_completed(self, future, cmap):
        # Ensure GUI updates are scheduled on the main thread
        result = future.result()

        self.binary_sauvola_inverted = result

        self.after(0, self.mask)

    def mask(self):
        self.processing_completed = False
        border_width = int(self.border_entry.get())
        masked_sauvola = self.binary_sauvola_inverted
        masked_sauvola[:border_width, :] = 0  # Top edge
        masked_sauvola[-border_width:, :] = 0  # Bottom edge
        masked_sauvola[:, :border_width] = 0  # Left edge
        masked_sauvola[:, -border_width:] = 0

        self.masked_sauvola = masked_sauvola

        self.distance()  # Compute distance transform

    def distance(self):
        self.processing_completed = False
        # Compute the distance transform
        self.distances = ndimage.distance_transform_edt(self.masked_sauvola)

        # Apply a threshold to focus on significant regions, as previously done
        threshold_value = float(self.threshold_value_entry.get()) * np.max(self.distances)
        self.thresholded_distance = np.where(self.distances > threshold_value, self.distances, 0)

        self.watershed()  # Apply watershed segmentation

    def watershed(self):
        self.processing_completed = False
        params = {
            'thresholded_distance': self.thresholded_distance,
            'distances': self.distances,
            'masked_sauvola': self.masked_sauvola,
            'min_distance': int(self.min_distance_entry.get())
        }

        future = self.executor.submit(watershed_processing, params)
        future.add_done_callback(lambda future: self.on_watershed_completed(future, 'nipy_spectral'))

    def on_watershed_completed(self, future, cmap):
        result = future.result()

        self.labels = result

        self.after(0, self.particle)

    def particle(self):
        self.processing_completed = False
        # Prepare parameters for particle analysis
        params = {
            'labels': self.labels,  # Assuming self.labels is available from previous processing
            'min_size': int(self.min_size_entry.get()),
            'min_roundness': float(self.min_roundness_entry.get())
        }

        # Submit the particle analysis task for parallel processing
        future = self.executor.submit(particle_processing, **params)
        future.add_done_callback(lambda future: self.on_particle_completed(future, 'gray'))
        # future.add_done_callback(self.on_particle_completed)

        self.processing_completed = True

        self.process_image()

    def on_particle_completed(self, future, cmap):
        # This method is called when the particle_processing task completes
        filtered_labels, boundaries = future.result()

        self.boundaries = boundaries
        self.filtered_labels = filtered_labels

    def overlay_current_image(self, xlim=None, ylim=None):
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

        # Update the Matplotlib figure for display

        self.plotting(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB), xlim=xlim, ylim=ylim)
        '''self.figure.clf()
        ax = self.figure.add_subplot(111)
        ax.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        ax.axis('off')
        self.canvas.draw()'''

    def process_all(self):
        #blurred = cv2.GaussianBlur(self.image_cv, (251, 251), 0)

        #self.img_back = cv2.subtract(blurred, self.image_cv)

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

        self.progress_bar.set(1 / 8)  # Update progress
        self.update_idletasks()

        self.img_back_sat = cv2.convertScaleAbs(self.img_back, alpha=float(self.a_entry.get()), beta=float(self.b_entry.get()))

        self.progress_bar.set(2 / 8)  # Update progress
        self.update_idletasks()

        # convert to 8-bit (grayscale)

        img_uint8 = (self.img_back_sat * 255).astype(np.uint8)

        # Apply Sauvola thresholding -> make a binary picture from the greyscale picture with local thresholding
        thresh_sauvola = threshold_sauvola(img_uint8, window_size=int(self.window_size_entry.get()),
                                           k=float(self.k_entry.get()))
        binary_sauvola = img_uint8 > thresh_sauvola

        # invert the picture to make particles white and background black -> needed for detection

        self.binary_sauvola_inverted = np.invert(binary_sauvola)

        self.progress_bar.set(3 / 8)  # Update progress
        self.update_idletasks()

        border_width = int(self.border_entry.get())
        masked_sauvola = self.binary_sauvola_inverted
        masked_sauvola[:border_width, :] = 0  # Top edge
        masked_sauvola[-border_width:, :] = 0  # Bottom edge
        masked_sauvola[:, :border_width] = 0  # Left edge
        masked_sauvola[:, -border_width:] = 0

        self.masked_sauvola = masked_sauvola

        self.progress_bar.set(4 / 8)  # Update progress
        self.update_idletasks()

        # Compute the distance transform
        self.distances = ndimage.distance_transform_edt(self.masked_sauvola)

        # Apply a threshold to focus on significant regions, as previously done
        threshold_value = float(self.threshold_value_entry.get()) * np.max(self.distances)
        self.thresholded_distance = np.where(self.distances > threshold_value, self.distances, 0)

        self.progress_bar.set(5 / 8)  # Update progress
        self.update_idletasks()

        coordinates = feature.peak_local_max(self.thresholded_distance, min_distance=int(self.min_distance_entry.get()),
                                             exclude_border=False)

        # Directly create unique markers for each peak without dilation
        markers = np.zeros_like(self.distances, dtype=int)
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)

        # Use the negative distance transform as input for watershed segmentation
        self.labels = segmentation.watershed(-self.thresholded_distance, markers, mask=self.masked_sauvola)

        self.progress_bar.set(6 / 8)  # Update progress
        self.update_idletasks()

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

        self.progress_bar.set(7 / 8)  # Final update, set to 100%
        self.update_idletasks()

        self.current_image = {'image_data': self.boundaries, 'process_type': 'particle'}
        self.overlay_current_image()

        self.progress_bar.set(1)  # Final update, set to 100%
        self.update_idletasks()

        self.process_image()

    def on_figure_click(self, event):
        if self.deselect_mode and event.inaxes:
            # Matplotlib's event handler provides the accurate x, y coordinates with respect to the image
            adjusted_x, adjusted_y = event.xdata, event.ydata

            # Check if a particle is clicked and remove it
            removed = self.remove_particle_if_clicked(int(adjusted_x), int(adjusted_y))
            if removed:
                self.redraw_image()
        else:
            pass

    def remove_particle_if_clicked(self, x, y):
        clicked_point = (int(y), int(x))
        for index, row in self.particles_df.iterrows():
            if clicked_point in row['Pixels']:
                # Get the label of the particle to remove
                label_to_remove = row['Label']
                # Set pixels with this label to 0 in the labels matrix
                self.filtered_labels[self.filtered_labels == label_to_remove] = 0
                # Remove particle from DataFrame
                self.particles_df.drop(index, inplace=True)
                self.particles_df.reset_index(drop=True, inplace=True)

                return True
        return False

    def redraw_image(self):
        # Ensure there are axes to work with
        if not hasattr(self, 'figure') or not self.figure.axes:
            return

        ax = self.figure.gca()  # Get current axes
        xlim = ax.get_xlim()  # Save current x-axis view limits
        ylim = ax.get_ylim()  # Save current y-axis view limits

        # Proceed with redrawing your image
        self.boundaries = segmentation.find_boundaries(self.filtered_labels)
        self.current_image = {'image_data': self.boundaries, 'process_type': 'particle'}
        self.overlay_current_image(xlim=xlim, ylim=ylim)

    def reset_view(self):
        self.overlay_current_image()
    def on_figure_click_select(self, event):
        if self.select_mode and event.inaxes:
            # Matplotlib's event handler provides the accurate x, y coordinates with respect to the image
            adjust_x, adjust_y = event.xdata, event.ydata

        added = self.add_particle_if_clicked(int(adjust_x), int(adjust_y))
        if added:
            self.redraw_image()

        else:
            pass

    def add_particle_if_clicked(self, x, y):
        # Convert the float x, y to integer pixel coordinates
        ix, iy = int(x), int(y)

        # Check if the coordinates are within the image boundaries
        if 0 <= ix < self.image_cv.shape[1] and 0 <= iy < self.image_cv.shape[0]:
            # Get the grayscale value at the clicked pixel
            gray_value = self.image_cv[iy, ix]

            ax = self.figure.gca()
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            # Convert axis limits to integer pixel indices
            # Note: This assumes the image directly maps to the data coordinates.
            x_min, x_max = int(xlim[0]), int(xlim[1])
            y_min, y_max = int(ylim[0]), int(ylim[1])

            # Crop the original grayscale image based on the zoomed area
            # Note: Matplotlib's y-axis starts from the top for images, so we invert y_min and y_max for cropping.
            cropped_image = self.image_cv[y_max:y_min, x_min:x_max]

            binary_image = np.where(cropped_image <= (gray_value * 1.5), 255, 0).astype(np.uint8)

            # Close holes inside particles if necessary
            binary_image = morphology.closing(binary_image, morphology.square(3))

            # Label connected components
            labeled_image = measure.label(binary_image)

            # Find the label of the particle at the click location (adjusting for crop)
            click_label = labeled_image[
                min(labeled_image.shape[0] - 1, iy - max(0, y_max)), min(labeled_image.shape[1] - 1,
                                                                         ix - max(0, x_min))]

            # Create a mask for the selected particle
            particle_mask = (labeled_image == click_label)
            boundaries = segmentation.find_boundaries(particle_mask)

            # Overlay boundaries on the cropped image
            overlay_image = cropped_image.copy()
            overlay_image[boundaries] = 255  # Make boundaries white

            new_label = self.filtered_labels.max() + 1

            # Prepare an empty mask for the whole image with the same shape as self.filtered_labels
            full_mask = np.zeros_like(self.filtered_labels)

            # Place the particle_mask into the full_mask at the correct location
            # Ensure the coordinates do not exceed the original image's dimensions
            full_mask[y_max:y_max + particle_mask.shape[0], x_min:x_min + particle_mask.shape[1]] = particle_mask

            # Assign the new label to the particle_mask region in full_mask
            full_mask[full_mask > 0] = new_label

            # Merge this new mask into self.filtered_labels
            self.filtered_labels[full_mask == new_label] = new_label

            self.process_image()

            return True  # Indicating a grayscale value was obtained and thresholding was attempted.
        else:
            return False  # Click was outside the image boundaries.

    def process_image(self):

        self.particles_df = pd.DataFrame(columns=['Label', 'Pixels'])

        props_filtered = measure.regionprops(self.filtered_labels)

        # Temporary list to hold data before creating a DataFrame
        new_rows = []

        # Iterate over each property object in props_filtered
        for prop in props_filtered:
            pixels = [(i[0], i[1]) for i in prop.coords]  # List of pixel coordinates for this particle
            new_rows.append({'Label': prop.label, 'Pixels': pixels})

        # Convert new_rows to a DataFrame
        new_rows_df = pd.DataFrame(new_rows)

        # If there are new rows to add
        if not new_rows_df.empty:
            # Concatenate the new rows with the existing DataFrame
            self.particles_df = pd.concat([self.particles_df, new_rows_df], ignore_index=True)

    def update_image_display(self):
        # Placeholder for redrawing the image and updating the GUI
        pass

    def process_data(self):

        if self.filepath:  # If a file is selected
            filename = os.path.basename(self.filepath)
            self.filenames.append(filename)

        props_filtered = measure.regionprops(self.filtered_labels)

        # Initialize lists to store area and diameter values
        areas_nm2 = []
        diameters_nm = []

        scale_factor = float(self.scale_factor_entry.get())

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

        if hasattr(self, 'df_particles'):
            last_index = self.df_particles['Index'].max()
            new_data = pd.DataFrame({
                'Index': np.arange(last_index + 1, last_index + 1 + len(areas_nm2)),
                'Area_nm2': areas_nm2,
                'Diameter_nm': diameters_nm
            })
            # Calculate diameter squared and cubed, add as new columns
            new_data['Diameter_nm_squared'] = new_data['Diameter_nm'] ** 2
            new_data['Diameter_nm_cubed'] = new_data['Diameter_nm'] ** 3

            self.df_particles = pd.concat([self.df_particles, new_data], ignore_index=True)


        else:
            self.df_particles = pd.DataFrame({
                'Index': np.arange(1, len(areas_nm2) + 1),
                'Area_nm2': areas_nm2,
                'Diameter_nm': diameters_nm
            })

            # Calculate diameter squared and cubed, add as new columns
            self.df_particles['Diameter_nm_squared'] = self.df_particles['Diameter_nm'] ** 2
            self.df_particles['Diameter_nm_cubed'] = self.df_particles['Diameter_nm'] ** 3

        # Sum up the diameter squared and cubed columns
        sum_diameter_squared = self.df_particles['Diameter_nm_squared'].sum()
        sum_diameter_cubed = self.df_particles['Diameter_nm_cubed'].sum()

        # Calculate the division of the sum of cubed by the sum of squared diameters
        self.surface_average = sum_diameter_cubed / sum_diameter_squared

        # Calculate the histogram
        bin_edges = np.arange(0, self.df_particles['Diameter_nm'].max() + 0.5, 0.5)
        self.hist, self.bins = np.histogram(self.df_particles['Diameter_nm'], bins=bin_edges, density=True)

        self.bin_centers = 0.5 * (self.bins[1:] + self.bins[:-1])
        bin_width = np.diff(self.bins)[0]  # Assuming uniform bin width

        # Extend the x values for fitting by one bin width at the start and end
        x_start = self.bins[0] - bin_width
        x_end = self.bins[-1] + bin_width
        self.x_extended = np.linspace(x_start, x_end, 100)

        # Fit a log-normal distribution to the diameters_nm data
        sigma, loc, scale = lognorm.fit(diameters_nm, floc=0)
        dist_extended = lognorm(s=sigma, loc=loc, scale=scale)

        # Generate PDF values for the extended x range
        self.pdf_fitted_extended = dist_extended.pdf(self.x_extended)

        # Find the maximum position (mode) of the log-normal distribution
        self.max_position = scale  # For a log-normal distribution

        # Calculate FWHM
        self.fwhm = np.exp(np.log(scale) + (sigma ** 2) / 2) * (np.exp(sigma ** 2) - 1) ** 0.5

        self.ecsa = 6 / ((self.surface_average * (10 ** -9)) * (float(self.density_entry.get()) * (10 ** 6)))

        # Clear the existing figure
        self.figure.clf()
        ax = self.figure.add_subplot(111)

        # Plot the histogram
        ax.bar(self.bin_centers, self.hist, width=np.diff(self.bins), alpha=0.5, label='Histogram')

        # Plot the fitted distribution
        ax.plot(self.x_extended, self.pdf_fitted_extended, 'r-', label='Log-normal fit')

        ax.set_xlabel('Diameter (nm)')
        ax.set_ylabel('Density')
        ax.set_title('Particle Size Distribution and Fit')
        ax.legend()
        self.figure.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        # Update the canvas to show the new plot
        self.canvas.draw()

        self.results = {
            'Filenames': "; ".join(self.filenames),  # Join filenames into a single string
            'Number_of_Paticles': self.df_particles['Index'].max(),
            'Average_Diameter_nm': self.df_particles['Diameter_nm'].mean(),
            'Average_Diameter_nm_std': self.df_particles['Diameter_nm'].std(),
            'Surface_Average': self.surface_average,
            'Expected_ECSA_m²/g': self.ecsa,
            'Max_Position_nm': self.max_position,
            'FWHM_nm': self.fwhm,
        }

        self.update_results_display(self.results)

    def clear_data(self):
        # does currently clear everything

        self.df_particles = pd.DataFrame()
        del self.df_particles
        self.particles_df = pd.DataFrame(columns=['Label', 'Pixels'])

        self.results = {
            'Filenames': '',
            'Number_of_Paticles': '',
            'Average_Diameter_nm': 0,
            'Average_Diameter_nm_std': 0,
            'Surface_Average': 0,
            'Expected_ECSA_m²/g': 0,
            'Max_Position_nm': 0,
            'FWHM_nm': 0,
        }

        self.update_results_display(self.results)

        self.results_labels['filenames'].configure(text=f"Filenames: ")
        self.results_labels['number_of_particles'].configure(text=f"Number of Particles: ")
        self.results_labels['average_diameter'].configure(text=f"Average Diameter: ")
        self.results_labels['surface_average'].configure(text=f"Surface Average: ")
        self.results_labels['expected_ecsa'].configure(text=f"Expected ECSA: ")
        self.results_labels['max_position'].configure(text=f"Max Position: ")
        self.results_labels['fwhm'].configure(text=f"FWHM: ")

    def save_results(self):
        initial_filename = "analysis_results"  # Default filename without extension
        if self.filenames:
            initial_filename = os.path.splitext(self.filenames[0])[0]

        # Ask for the base path (without specifying filetypes to allow for a generic base path)
        save_base_path = filedialog.asksaveasfilename(initialfile=initial_filename, defaultextension="")

        # Check if the user provided a path, if not return to avoid errors
        if not save_base_path:
            return

        # Construct specific filenames based on the base path
        save_path = f"{save_base_path}_particle_analysis_plot.txt"
        save_path2 = f"{save_base_path}_particle_analysis_results.txt"
        save_path3 = f"{save_base_path}_boundaries.jpg"
        save_path4 = f"{save_base_path}_labels.jpg"

        histogram_data = pd.DataFrame({
            'Bin_centers': self.bin_centers,
            'Hist_counts': self.hist * np.diff(self.bins),  # Total count per bin
            'Hist_density': self.hist,  # Normalized counts per bin
        })
        fit_data = pd.DataFrame({
            'X_axis': self.x_extended,
            'Y_axis': self.pdf_fitted_extended,
        })

        # Concatenate histogram and fit data
        combined_data = pd.concat([histogram_data, fit_data], axis=1)

        # Prepare results dictionary with summary metrics and filenames
        '''results = {
            'Filenames': "; ".join(self.filenames),  # Join filenames into a single string
            'Average_Diameter_nm': self.df_particles['Diameter_nm'].mean(),
            'Surface_Average': self.surface_average,
            'Max_Position_nm': self.max_position,
            'FWHM_nm': self.fwhm,
        }'''
        results = self.results
        # Convert results dictionary to DataFrame
        results_df = pd.DataFrame([results])

        # Combine all data into one DataFrame
        final_combined_data = pd.concat([self.df_particles, combined_data], axis=1)

        boundaries_uint8 = (self.boundaries * 255).astype(np.uint8)
        # Save the boundaries image
        cv2.imwrite(save_path3, boundaries_uint8)

        labels_uint8 = (self.filtered_labels > 0).astype(np.uint8) * 255
        # Save the filtered_labels image
        cv2.imwrite(save_path4, labels_uint8)

        # Save combined data to the specified file path
        final_combined_data.to_csv(save_path, index=False, sep='\t')
        results_df.to_csv(save_path2, index=False, sep='\t')

        self.clear_data()

    def on_close(self):
        # Properly shutdown the executor on application close
        self.executor.shutdown()
        self.destroy()
        if self.cid is not None:
            self.figure.canvas.mpl_disconnect(self.cid)


# Run the app
if __name__ == '__main__':
    app = ImageAnalyzerApp()
    app.mainloop()

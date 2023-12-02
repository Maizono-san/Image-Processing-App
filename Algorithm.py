from PIL import Image
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter, uniform_filter, convolve, median_filter, gaussian_filter
import tkinter.messagebox as messagebox
import cv2
from skimage import filters
from statistics import mode


def open_and_convert(input_image_path):
    input_image = Image.open(input_image_path)

    gray_image = input_image.convert("L")

    return gray_image


class ImageEnhancement:
    @staticmethod
    def apply_negative_image(input_image_path):
        gray_image = open_and_convert(input_image_path)

        # Inverting color
        output_image = Image.eval(gray_image, lambda x: 255 - x)

        return output_image

    @staticmethod
    def apply_thresholding_algorithm(input_image_path, threshold=50):
        gray_image = open_and_convert(input_image_path)

        # Apply thresholding
        thresholded_image = gray_image.point(lambda x: 0 if x < threshold else 255, '1')

        return thresholded_image

    @staticmethod
    def apply_logarithmic_transformation(input_image_path):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image)

        # Apply logarithmic transformation
        c = 255 / np.log(1 + np.max(image_array))
        log_transformed_image = c * (np.log(image_array + 0.99))

        # Normalize the values to be in the range [0, 255]
        log_transformed_image = (log_transformed_image / np.max(log_transformed_image)) * 255

        # Convert the NumPy array back to an image
        log_transformed_image = Image.fromarray(log_transformed_image.astype('uint8'))

        return log_transformed_image

    @staticmethod
    def apply_power_law_transformation(input_image_path, gamma):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image)

        # Apply power-law transformation
        power_transformed_image = np.power(image_array / 255.0, gamma) * 255.0

        # Convert the NumPy array back to an image
        power_transformed_image = Image.fromarray(power_transformed_image.astype('uint8'))

        return power_transformed_image

    @staticmethod
    def apply_minimum_filter(input_image_path, kernel_size):
        # Load the input image
        if kernel_size <= 0:
            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")
        else:
            gray_image = open_and_convert(input_image_path)

            # Convert the image to a NumPy array for faster processing
            image_array = np.array(gray_image)

            # Apply the minimum filter
            filtered_image = minimum_filter(image_array, size=kernel_size, mode='constant', cval=0)

            # Convert the NumPy array back to an image
            filtered_image = Image.fromarray(filtered_image.astype('uint8'))

            return filtered_image

    @staticmethod
    def apply_maximum_filter(input_image_path, kernel_size):
        if kernel_size <= 0:
            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")
        else:
            gray_image = open_and_convert(input_image_path)

            # Convert the image to a NumPy array for faster processing
            image_array = np.array(gray_image)

            # Apply the maximum filter
            filtered_image = maximum_filter(image_array, size=kernel_size, mode='constant', cval=0)

            # Convert the NumPy array back to an image
            filtered_image = Image.fromarray(filtered_image.astype('uint8'))

            return filtered_image

    @staticmethod
    def apply_simple_average_filter(input_image_path, kernel_size):
        if kernel_size <= 0:
            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")
        else:
            gray_image = open_and_convert(input_image_path)

            # Convert the image to a NumPy array for faster processing
            image_array = np.array(gray_image, dtype=np.float64)

            # Apply the Simple Average Filter
            filtered_image = uniform_filter(image_array, size=kernel_size, mode='constant', cval=0)

            # Round
            rounded_image = np.round(filtered_image)

            # Convert the NumPy array back to an image
            rounded_image = Image.fromarray(rounded_image.astype('uint8'))

            return rounded_image

    @staticmethod
    def apply_weighted_average_filter(input_image_path, kernel):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image, dtype=np.float64)

        # Apply the Weighted Average Filter using convolution
        filtered_image = convolve(image_array, kernel, mode='constant', cval=0)

        # Round
        rounded_image = np.round(filtered_image)

        # Convert the NumPy array back to an image
        rounded_image = Image.fromarray(rounded_image.astype('uint8'))

        return rounded_image

    @staticmethod
    def apply_median_filter(input_image_path, kernel_size):
        if kernel_size <= 0:
            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")
        else:
            gray_image = open_and_convert(input_image_path)

            # Convert the image to a NumPy array for faster processing
            image_array = np.array(gray_image)

            # Apply median filter
            filtered_image = median_filter(image_array, size=kernel_size, mode='constant', cval=0)

            # Convert the NumPy array back to an image
            filtered_image = Image.fromarray(filtered_image.astype('uint8'))

            return filtered_image

    @staticmethod
    def apply_k_nearest_mean_filter(input_image_path, filter_function, parameters):
        if parameters[0] <= 0 or parameters[1] <= 0 or parameters[2] <= 0:
            messagebox.showwarning("Lỗi", "Giá trị không hợp lệ. ")
        else:
            gray_image = open_and_convert(input_image_path)

            # Convert the image to a NumPy array for faster processing
            image_array = np.array(gray_image)

            # Apply the custom filter function
            output_image = filter_function(image_array, parameters)

            # Convert the NumPy array back to an image
            output_image = Image.fromarray(output_image.astype('uint8'))

            return output_image


# Thêm hàm k_nearest_mean_filter bên ngoài class ImageProcessor
def k_nearest_mean_filter(image_array, parameters):
    k, kernel_size, thresh_hold = parameters[0], parameters[1], parameters[2]
    original_data = image_array.copy()

    # k nearest neighbour mean filter algorithm
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            neighbour = []
            center_value = original_data[i, j]
            for x in range(i - kernel_size // 2, i + kernel_size // 2 + 1):
                for y in range(j - kernel_size // 2, j + kernel_size // 2 + 1):
                    if 0 <= x < image_array.shape[0] and 0 <= y < image_array.shape[1]:
                        neighbour.append(original_data[x, y])
                    else:
                        neighbour.append(0)
            neighbour.sort(key=lambda val: abs(np.subtract(val, center_value)))
            neighbour = np.array(neighbour)[:k]
            k_nearest_mean = np.round(np.mean(neighbour))
            if abs(center_value - k_nearest_mean) > thresh_hold:
                image_array[i, j] = k_nearest_mean

    return image_array


class EdgeDetection:
    @staticmethod
    def apply_1d_operator(input_image_path):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image)

        # 1D Operator Kernels
        mask1d_x = np.array([[-1], [1]])
        mask1d_y = np.array([[-1, 1]])

        # Apply 1D Operator
        gradient_x = convolve(image_array, mask1d_x, mode='constant', cval=0)
        gradient_y = convolve(image_array, mask1d_y, mode='constant', cval=0)

        # Combine the gradients to get the magnitude
        gradient_magnitude = np.round(np.sqrt(gradient_x ** 2 + gradient_y ** 2))

        # Normalize the values to be in the range [0, 255]
        gradient_magnitude = np.round((gradient_magnitude / np.max(gradient_magnitude)) * 255)

        # Convert the NumPy array back to an image
        gradient_image = Image.fromarray(gradient_magnitude.astype('uint8'))

        return gradient_image

    @staticmethod
    def apply_roberts_operator(input_image_path):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image)

        # Roberts Operator Kernels
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])

        # Apply Roberts Operator
        gradient_x = convolve(image_array, roberts_x, mode='constant', cval=0)
        gradient_y = convolve(image_array, roberts_y, mode='constant', cval=0)

        # Combine the gradients to get the magnitude
        gradient_magnitude = np.round(np.sqrt(gradient_x**2 + gradient_y**2))

        # Normalize the values to be in the range [0, 255]
        gradient_magnitude = np.round((gradient_magnitude / np.max(gradient_magnitude)) * 255)

        # Convert the NumPy array back to an image
        gradient_image = Image.fromarray(gradient_magnitude.astype('uint8'))

        return gradient_image

    @staticmethod
    def apply_prewitt_operator(input_image_path):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image)

        # Prewitt Operator Kernels
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        # Apply Prewitt Operator
        gradient_x = convolve(image_array, prewitt_x, mode='constant', cval=0)
        gradient_y = convolve(image_array, prewitt_y, mode='constant', cval=0)

        # Combine the gradients to get the magnitude
        gradient_magnitude = np.round(np.sqrt(gradient_x ** 2 + gradient_y ** 2))

        # Normalize the values to be in the range [0, 255]
        gradient_magnitude = np.round((gradient_magnitude / np.max(gradient_magnitude)) * 255)

        # Convert the NumPy array back to an image
        gradient_image = Image.fromarray(gradient_magnitude.astype('uint8'))

        return gradient_image

    @staticmethod
    def apply_sobel_operator(input_image_path):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image)

        # Sobel Operator Kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Apply Sobel Operator
        gradient_x = convolve(image_array, sobel_x, mode='constant', cval=0)
        gradient_y = convolve(image_array, sobel_y, mode='constant', cval=0)

        # Combine the gradients to get the magnitude
        gradient_magnitude = np.round(np.sqrt(gradient_x ** 2 + gradient_y ** 2))

        # Normalize the values to be in the range [0, 255]
        gradient_magnitude = np.round((gradient_magnitude / np.max(gradient_magnitude)) * 255)

        # Convert the NumPy array back to an image
        gradient_image = Image.fromarray(gradient_magnitude.astype('uint8'))

        return gradient_image

    @staticmethod
    def apply_laplacian_operator(input_image_path):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image)

        # Laplacian Operator Kernel
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

        # Apply Laplacian Operator
        laplacian_image = convolve(image_array, laplacian_kernel, mode='constant', cval=0)

        # Normalize the values to be in the range [0, 255]
        laplacian_image = np.round((laplacian_image / np.max(laplacian_image)) * 255)

        # Convert the NumPy array back to an image
        laplacian_image = Image.fromarray(laplacian_image.astype('uint8'))

        return laplacian_image

    @staticmethod
    def apply_canny_operator1(input_image_path, sigma=1.0, low_threshold=20, high_threshold=150):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image)

        # Apply Gaussian smoothing to reduce noise
        smoothed_image = gaussian_filter(image_array, sigma=sigma)

        # Calculate gradients using Sobel operators
        gradient_x = convolve(smoothed_image, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), mode='constant', cval=0)
        gradient_y = convolve(smoothed_image, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]), mode='constant', cval=0)

        # Calculate gradient magnitude and direction
        gradient_magnitude = np.round(np.sqrt(gradient_x ** 2 + gradient_y ** 2))
        gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

        # Normalize the values to be in the range [0, 255]
        gradient_magnitude = np.round((gradient_magnitude / np.max(gradient_magnitude)) * 255)

        # Non-maximum suppression
        non_max_suppressed = EdgeDetection.non_maximum_suppression(gradient_magnitude, gradient_direction)

        # Double thresholding and edge tracking by hysteresis
        edge_image = EdgeDetection.hysteresis_thresholding(non_max_suppressed, low_threshold, high_threshold)

        # Convert the NumPy array back to an image
        edge_image = Image.fromarray(edge_image.astype('uint8'))

        return edge_image

    @staticmethod
    def non_maximum_suppression(gradient_magnitude, gradient_direction):
        # Create an array for the non-maximum suppression result
        result = np.zeros_like(gradient_magnitude)

        for i in range(1, gradient_magnitude.shape[0] - 1):
            for j in range(1, gradient_magnitude.shape[1] - 1):
                angle = gradient_direction[i, j]

                # Determine the indices of neighboring pixels
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    neighbors = [(i, j + 1), (i, j - 1)]
                elif 22.5 <= angle < 67.5:
                    neighbors = [(i - 1, j + 1), (i + 1, j - 1)]
                elif 67.5 <= angle < 112.5:
                    neighbors = [(i - 1, j), (i + 1, j)]
                elif 112.5 <= angle < 157.5:
                    neighbors = [(i - 1, j - 1), (i + 1, j + 1)]

                # Compare the current pixel with its neighbors
                if all(gradient_magnitude[i, j] >= gradient_magnitude[n[0], n[1]] for n in neighbors):
                    result[i, j] = gradient_magnitude[i, j]

        return result

    @staticmethod
    def hysteresis_thresholding(image, low_threshold, high_threshold):
        strong_edges = (image >= high_threshold)
        weak_edges = (low_threshold <= image) & (image < high_threshold)

        # Use depth-first search to track weak edges connected to strong edges
        visited = np.zeros_like(image)
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                if strong_edges[i, j] and not visited[i, j]:
                    EdgeDetection.depth_first_search(image, visited, strong_edges, weak_edges, i, j)

        # Assign final edge values
        result = np.zeros_like(image)
        result[strong_edges] = 255
        result[weak_edges] = 128  # You can adjust the intensity for weak edges if needed

        return result

    @staticmethod
    def depth_first_search(image, visited, strong_edges, weak_edges, i, j):
        stack = [(i, j)]

        while stack:
            current_i, current_j = stack.pop()

            if not (0 <= current_i < image.shape[0] and 0 <= current_j < image.shape[1]):
                continue

            if visited[current_i, current_j] or not weak_edges[current_i, current_j]:
                continue

            visited[current_i, current_j] = 1
            strong_edges[current_i, current_j] = 1

            for x in range(current_i - 1, current_i + 2):
                for y in range(current_j - 1, current_j + 2):
                    stack.append((x, y))

        return strong_edges

    @staticmethod
    def apply_canny_operator2(input_image_path, low_threshold=20, high_threshold=150):
        # Load the input image
        input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

        # Apply Gaussian smoothing to reduce noise
        smoothed_image = cv2.GaussianBlur(input_image, (5, 5), 0)

        # Apply Canny edge detection
        edge_image = cv2.Canny(smoothed_image, low_threshold, high_threshold)

        # Convert the NumPy array back to an image
        edge_image = Image.fromarray(edge_image.astype('uint8'))

        return edge_image


class Segmentation:
    @staticmethod
    def apply_otsu_segmentation(input_image_path):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image)

        # Apply Otsu's segmentation
        threshold_value = filters.threshold_otsu(image_array)
        segmented_image = gray_image.point(lambda x: 0 if x < threshold_value else 255, '1')

        return segmented_image

    @staticmethod
    def apply_isodata_segmentation(input_image_path):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image)

        # Apply Isodata segmentation
        threshold_value = Segmentation.isodata_threshold(image_array)
        segmented_image = gray_image.point(lambda x: 0 if x < threshold_value else 255, '1')

        return segmented_image

    @staticmethod
    def isodata_threshold(image_array):
        # Initial threshold value (average intensity)
        threshold_value = np.mean(image_array)

        while True:
            # Calculate the mean intensities of pixels below and above the threshold
            below_threshold = image_array[image_array <= threshold_value]
            above_threshold = image_array[image_array > threshold_value]

            if len(below_threshold) == 0 or len(above_threshold) == 0:
                break

            mean_below = np.mean(below_threshold)
            mean_above = np.mean(above_threshold)

            # Update the threshold value
            new_threshold_value = 0.5 * (mean_below + mean_above)

            # Check for convergence
            if abs(threshold_value - new_threshold_value) < 0.5:
                break

            threshold_value = new_threshold_value

        return threshold_value

    @staticmethod
    def apply_background_symmetry_algorithm(input_image_path):
        gray_image = open_and_convert(input_image_path)

        # Convert the image to a NumPy array for faster processing
        image_array = np.array(gray_image)

        # Apply background symmetry algorithm
        symmetry_threshold = Segmentation.calculate_background_symmetry(image_array)
        segmented_image = gray_image.point(lambda x: 0 if x < symmetry_threshold else 255, '1')

        return segmented_image

    @staticmethod
    def calculate_background_symmetry(image_array):
        # Flatten the image array
        flattened_array = image_array.flatten()

        # Calculate the mode (most frequent pixel intensity)
        maxp = int(mode(flattened_array))

        # Calculate the total frequency
        total_frequency = len(flattened_array)

        # Calculate the desired percentile
        percentile = 0.95
        p_percent = round(total_frequency * percentile)

        # Calculate cumulative frequency to find p
        cumulative_frequency = 0
        p = 0
        for level in sorted(np.unique(flattened_array)):
            cumulative_frequency += len(flattened_array[flattened_array == level])
            if cumulative_frequency == p_percent:
                p = level
                break
            elif cumulative_frequency > p_percent:
                p = level - 1
                break

        symmetry_threshold = maxp - (p - maxp)

        # Normalization
        if symmetry_threshold > 255:
            symmetry_threshold = 255
        else:
            symmetry_threshold = symmetry_threshold

        return symmetry_threshold

    @staticmethod
    def apply_triangle_algorithm(input_image_path):
        gray_image = open_and_convert(input_image_path)

        # Apply Triangle Algorithm
        threshold_value = filters.threshold_triangle(np.array(gray_image))
        segmented_image = gray_image.point(lambda x: 0 if x < threshold_value else 255, '1')

        return segmented_image


class MorphologicalProcessing:
    @staticmethod
    def apply_erosion(input_image_path, kernel_size):
        if kernel_size <= 0:
            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")
        else:
            # Read the input image using cv2
            input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            # Define the kernel for Erosion using cv2
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Apply Erosion using cv2.erode
            eroded_image = cv2.erode(input_image, kernel, iterations=1)

            # Convert the NumPy array back to an image
            eroded_image = Image.fromarray(eroded_image.astype('uint8'))

            return eroded_image

    @staticmethod
    def apply_dilation(input_image_path, kernel_size):
        if kernel_size <= 0:
            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")
        else:
            # Read the input image using cv2
            input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            # Define the kernel for Dilation using cv2
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Apply Dilation using cv2.dilate
            dilated_image = cv2.dilate(input_image, kernel, iterations=1)

            # Convert the NumPy array back to an image
            dilated_image = Image.fromarray(dilated_image.astype('uint8'))

            return dilated_image

    @staticmethod
    def apply_opening(input_image_path, kernel_size):
        if kernel_size <= 0:
            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")
        else:
            # Read the input image using cv2
            input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            # Define the kernel for Opening using cv2
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Apply Opening using cv2.morphologyEx
            opened_image = cv2.morphologyEx(input_image, cv2.MORPH_OPEN, kernel)

            # Convert the NumPy array back to an image
            opened_image = Image.fromarray(opened_image.astype('uint8'))

            return opened_image

    @staticmethod
    def apply_closing(input_image_path, kernel_size):
        if kernel_size <= 0:
            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")
        else:
            # Read the input image using cv2
            input_image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            # Define the kernel for Closing using cv2
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Apply Closing using cv2.morphologyEx
            closed_image = cv2.morphologyEx(input_image, cv2.MORPH_CLOSE, kernel)

            # Convert the NumPy array back to an image
            closed_image = Image.fromarray(closed_image.astype('uint8'))

            return closed_image

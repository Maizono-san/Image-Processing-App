import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from Algorithm import ImageEnhancement, k_nearest_mean_filter, EdgeDetection, Segmentation, MorphologicalProcessing
import tkinter.messagebox as messagebox
import numpy as np


class HomePage:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing App")

        window_width = 400
        window_height = 200
        screen_width = master.winfo_screenwidth()
        screen_height = master.winfo_screenheight()
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        master.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        master.resizable(False, False)
        master.configure(bg='#363636')

        self.icon = tk.PhotoImage(file="img/icon.png")
        master.iconphoto(True, self.icon)

        self.algorithm_label = tk.Label(master, text="Chọn cách xử lý ảnh", font=("Segoe UI", 20), fg='white',
                                        bg='#363636')
        self.algorithm_label.pack(pady=10)

        self.algorithm_combobox = ttk.Combobox(master, values=["Tăng cường ảnh", "Phát hiện biên", "Phân vùng ảnh",
                                                               "Xử lý hình thái"],
                                               font=("Segoe UI", 13))

        self.algorithm_combobox.option_add('*TCombobox*Listbox.font', ("Segoe UI", 13))

        self.algorithm_combobox.pack(pady=10)

        self.button = tk.PhotoImage(file="img/blue_button.png")
        self.select_button = tk.Button(master, text="Chọn", image=self.button, command=self.select_algorithm,
                                       fg='white', bg='#363636', compound='center', font=("Segoe UI", 15),
                                       cursor='hand2', activeforeground='white', activebackground='#363636',
                                       relief=tk.GROOVE, borderwidth=0)
        self.select_button.pack(pady=20)

    def select_algorithm(self):
        selected_algorithm = self.algorithm_combobox.get()
        if selected_algorithm == "Tăng cường ảnh":
            enhancement_window = tk.Toplevel(self.master)
            ImageEnhancementPage(enhancement_window, self.master)
        elif selected_algorithm == "Phát hiện biên":
            enhancement_window = tk.Toplevel(self.master)
            EdgeDetectionPage(enhancement_window, self.master)
        elif selected_algorithm == "Phân vùng ảnh":
            enhancement_window = tk.Toplevel(self.master)
            SegmentationPage(enhancement_window, self.master)
        elif selected_algorithm == "Xử lý hình thái":
            enhancement_window = tk.Toplevel(self.master)
            MorphologicalProcessingPage(enhancement_window, self.master)
        else:
            messagebox.showwarning("Lỗi", "Vui lòng nhập chọn cách xử lý ảnh")


class ImageEnhancementPage:
    def __init__(self, master, home_window):
        self.master = master
        self.home_window = home_window
        home_window.withdraw()

        self.master.title("Tăng cường ảnh")
        self.master.configure(bg='#363636')

        window_width = master.winfo_screenwidth()
        window_height = 800
        self.master.geometry(f"{window_width}x{window_height}+0+0")

        self.master.resizable(False, False)

        self.algorithm_label = tk.Label(self.master, text="Chọn thuật toán tăng cường ảnh", font=("Segoe UI", 20),
                                        fg='white', bg='#363636')
        self.algorithm_label.pack(pady=10)

        self.button = tk.PhotoImage(file="img/red_button.png")
        self.back_button = tk.Button(self.master, text="Trở về", command=self.go_back_home, image=self.button,
                                     fg='white', bg='#363636', compound='center', font=("Segoe UI", 15),
                                     cursor='hand2', activeforeground='white', activebackground='#363636',
                                     relief=tk.GROOVE, borderwidth=0)
        self.back_button.place(x=10, y=30)

        self.algorithm_combobox = ttk.Combobox(self.master, values=["Âm bản", "Phân ngưỡng", "Biến đổi logarit",
                                                                    "Biến đổi hàm mũ", "Bộ lọc min", "Bộ lọc max",
                                                                    "Bộ lọc trung bình đơn giản",
                                                                    "Bộ lọc trung bình trọng số",
                                                                    "Bộ lọc trung bình k giá trị gần nhất",
                                                                    "Bộ lọc trung vị"], font=("Segoe UI", 13))

        self.algorithm_combobox.option_add('*TCombobox*Listbox.font', ("Segoe UI", 13))
        self.algorithm_combobox.pack(pady=0)

        # Create Frame
        self.left_frame = tk.Frame(self.master, width=window_width // 2, height=window_height, bg='#363636')
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.master, width=window_width // 2, height=window_height, bg='#363636')
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.big_button = tk.PhotoImage(file="img/big_button.png")
        self.input_button = tk.Button(self.left_frame, text="Chọn ảnh đầu vào", command=self.select_input_image,
                                      image=self.big_button, fg='white', bg='#363636', compound='center',
                                      font=("Segoe UI", 15), cursor='hand2', activeforeground='white',
                                      activebackground='#363636', relief=tk.GROOVE, borderwidth=0)
        self.input_button.pack(pady=20)

        self.output_button = tk.Button(self.right_frame, text="Hiển thị ảnh đầu ra", command=self.show_output_image,
                                       image=self.big_button, fg='white', bg='#363636', compound='center',
                                       font=("Segoe UI", 15), cursor='hand2', activeforeground='white',
                                       activebackground='#363636', relief=tk.GROOVE, borderwidth=0)
        self.output_button.pack(pady=20)

        self.input_image_path = ""
        self.output_image_path = ""

        # Create Canvas
        self.input_canvas = tk.Canvas(self.left_frame, bg="white", width=window_width // 2, height=window_height // 2)
        self.input_canvas.pack()
        self.output_canvas = tk.Canvas(self.right_frame, bg="white", width=window_width // 2, height=window_height // 2)
        self.output_canvas.pack()

        self.gamma_label = tk.Label(self.right_frame, text="Giá trị gamma", font=("Segoe UI", 12), fg='white',
                                    bg='#363636')
        self.gamma_label.pack(pady=2)
        self.gamma_entry = tk.Entry(self.right_frame, font=("Segoe UI", 12))
        self.gamma_entry.pack(pady=2)
        self.gamma_label.pack_forget()
        self.gamma_entry.pack_forget()

        self.kernel_size_label = tk.Label(self.right_frame, text="Kích thước bộ lọc", font=("Segoe UI", 12), fg='white',
                                          bg='#363636')
        self.kernel_size_label.pack(pady=2)
        self.kernel_size_entry = tk.Entry(self.right_frame, font=("Segoe UI", 12))
        self.kernel_size_entry.pack(pady=2)
        self.kernel_size_label.pack_forget()
        self.kernel_size_entry.pack_forget()

        self.k_label = tk.Label(self.right_frame, text="Giá trị k", font=("Segoe UI", 12), fg='white', bg='#363636')
        self.k_label.pack(pady=2)
        self.k_entry = tk.Entry(self.right_frame, font=("Segoe UI", 12))
        self.k_entry.pack(pady=2)
        self.k_label.pack_forget()
        self.k_entry.pack_forget()

        self.threshold_label = tk.Label(self.right_frame, text="Ngưỡng", font=("Segoe UI", 12), fg='white',
                                        bg='#363636')
        self.threshold_label.pack(pady=2)
        self.threshold_entry = tk.Entry(self.right_frame, font=("Segoe UI", 12))
        self.threshold_entry.pack(pady=2)
        self.threshold_label.pack_forget()
        self.threshold_entry.pack_forget()

        self.algorithm_combobox.bind("<<ComboboxSelected>>", self.update_entry_visibility)

    def go_back_home(self):
        self.master.destroy()
        self.home_window.deiconify()

    def update_entry_visibility(self, event):
        selected_algorithm = self.algorithm_combobox.get()
        if selected_algorithm == "Biến đổi hàm mũ":
            self.gamma_label.pack(pady=2)
            self.gamma_entry.pack(pady=2)
            self.kernel_size_label.pack_forget()
            self.kernel_size_entry.pack_forget()
            self.k_label.pack_forget()
            self.k_entry.pack_forget()
            self.threshold_label.pack_forget()
            self.threshold_entry.pack_forget()
        elif selected_algorithm in ["Bộ lọc min", "Bộ lọc max", "Bộ lọc trung bình đơn giản", "Bộ lọc trung vị"]:
            self.kernel_size_label.pack(pady=2)
            self.kernel_size_entry.pack(pady=2)
            self.gamma_label.pack_forget()
            self.gamma_entry.pack_forget()
            self.k_label.pack_forget()
            self.k_entry.pack_forget()
            self.threshold_label.pack_forget()
            self.threshold_entry.pack_forget()
        elif selected_algorithm == "Bộ lọc trung bình k giá trị gần nhất":
            self.k_label.pack(pady=2)
            self.k_entry.pack(pady=2)
            self.kernel_size_label.pack(pady=2)
            self.kernel_size_entry.pack(pady=2)
            self.threshold_label.pack(pady=2)
            self.threshold_entry.pack(pady=2)
            self.gamma_label.pack_forget()
            self.gamma_entry.pack_forget()
        else:
            self.gamma_label.pack_forget()
            self.gamma_entry.pack_forget()
            self.kernel_size_label.pack_forget()
            self.kernel_size_entry.pack_forget()
            self.k_label.pack_forget()
            self.k_entry.pack_forget()
            self.threshold_label.pack_forget()
            self.threshold_entry.pack_forget()

    def select_input_image(self):
        file_path = filedialog.askopenfilename(title="Chọn ảnh đầu vào",
                                               filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.input_image_path = file_path
            print(f"Ảnh đầu vào đã chọn: {self.input_image_path}")

            # Show image
            self.show_input_image()

    def show_input_image(self):
        if self.input_image_path:
            # Show image
            input_image = Image.open(self.input_image_path)
            input_image = self.resize_image(input_image, self.input_canvas.winfo_width(),
                                            self.input_canvas.winfo_height())
            self.input_photo = ImageTk.PhotoImage(input_image)
            self.input_canvas.create_image(0, 0, anchor="nw", image=self.input_photo)

    def show_output_image(self):
        if self.input_image_path:
            processor = ImageEnhancement()
            if self.algorithm_combobox.get() == "":
                messagebox.showwarning("Lỗi", "Vui lòng chọn thuật toán tăng cường ảnh trước.")
            else:
                output_image = None
                if self.algorithm_combobox.get() == "Âm bản":
                    output_image = processor.apply_negative_image(self.input_image_path)

                elif self.algorithm_combobox.get() == "Phân ngưỡng":
                    output_image = processor.apply_thresholding_algorithm(self.input_image_path)

                elif self.algorithm_combobox.get() == "Biến đổi logarit":
                    output_image = processor.apply_logarithmic_transformation(self.input_image_path)

                elif self.algorithm_combobox.get() == "Biến đổi hàm mũ":
                    gamma_value = self.gamma_entry.get()
                    if gamma_value == "":
                        messagebox.showwarning("Lỗi", "Vui lòng nhập gamma.")
                    else:
                        try:
                            output_image = processor.apply_power_law_transformation(self.input_image_path,
                                                                                    gamma=float(gamma_value))
                        except ValueError:
                            messagebox.showwarning("Lỗi", "Giá trị gamma không hợp lệ. Vui lòng nhập một số.")

                elif self.algorithm_combobox.get() == "Bộ lọc min":
                    kernel_size_value = self.kernel_size_entry.get()
                    if kernel_size_value == "":
                        messagebox.showwarning("Lỗi", "Vui lòng nhập kích thước bộ lọc.")
                    else:
                        try:
                            output_image = processor.apply_minimum_filter(self.input_image_path,
                                                                          kernel_size=int(kernel_size_value))
                        except ValueError:
                            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")

                elif self.algorithm_combobox.get() == "Bộ lọc max":
                    kernel_size_value = self.kernel_size_entry.get()
                    if kernel_size_value == "":
                        messagebox.showwarning("Lỗi", "Vui lòng nhập kích thước bộ lọc.")
                    else:
                        try:
                            output_image = processor.apply_maximum_filter(self.input_image_path,
                                                                          kernel_size=int(kernel_size_value))
                        except ValueError:
                            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")

                elif self.algorithm_combobox.get() == "Bộ lọc trung bình đơn giản":
                    kernel_size_value = self.kernel_size_entry.get()
                    if kernel_size_value == "":
                        messagebox.showwarning("Lỗi", "Vui lòng nhập kích thước bộ lọc.")
                    else:
                        try:
                            output_image = processor.apply_simple_average_filter(self.input_image_path,
                                                                                 kernel_size=int(kernel_size_value))
                        except ValueError:
                            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")

                elif self.algorithm_combobox.get() == "Bộ lọc trung bình trọng số":
                    # Define your custom weight kernel
                    custom_kernel = np.array([
                        [1/16, 2/16, 1/16],
                        [2/16, 4/16, 2/16],
                        [1/16, 2/16, 1/16]
                    ])
                    output_image = processor.apply_weighted_average_filter(self.input_image_path, kernel=custom_kernel)

                elif self.algorithm_combobox.get() == "Bộ lọc trung vị":
                    kernel_size_value = self.kernel_size_entry.get()
                    if kernel_size_value == "":
                        messagebox.showwarning("Lỗi", "Vui lòng nhập kích thước bộ lọc.")
                    else:
                        try:
                            output_image = processor.apply_median_filter(self.input_image_path,
                                                                         kernel_size=int(kernel_size_value))
                        except ValueError:
                            messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")

                elif self.algorithm_combobox.get() == "Bộ lọc trung bình k giá trị gần nhất":
                    k_value = self.k_entry.get()
                    kernel_size_value = self.kernel_size_entry.get()
                    threshold_value = self.threshold_entry.get()

                    if not k_value or not kernel_size_value or not threshold_value:
                        messagebox.showwarning("Lỗi", "Vui lòng nhập giá trị cho k, kernel_size, và threshold.")
                    else:
                        try:
                            parameters = [int(k_value), int(kernel_size_value), int(threshold_value)]
                            output_image = processor.apply_k_nearest_mean_filter(self.input_image_path,
                                                                                 k_nearest_mean_filter,
                                                                                 parameters)
                        except ValueError:
                            messagebox.showwarning("Lỗi", "Giá trị không hợp lệ. ")

                # Save image
                if output_image is not None:
                    self.output_image_path = "output_image.png"
                    output_image.save(self.output_image_path)

                    # Show image
                    output_image = Image.open(self.output_image_path)
                    output_image = self.resize_image(output_image, self.output_canvas.winfo_width(),
                                                     self.output_canvas.winfo_height())
                    self.output_photo = ImageTk.PhotoImage(output_image)
                    self.output_canvas.create_image(0, 0, anchor="nw", image=self.output_photo)

        else:
            messagebox.showwarning("Lỗi", "Vui lòng chọn ảnh đầu vào trước.")

    @staticmethod
    def resize_image(image, width, height):
        if image.width > width or image.height > height:
            return image.resize((width, height), Image.LANCZOS)
        else:
            return image.resize((image.width, image.height), Image.LANCZOS)


class EdgeDetectionPage:
    def __init__(self, master, home_window):
        self.master = master
        self.home_window = home_window
        home_window.withdraw()

        self.master.title("Phát hiện biên")
        self.master.configure(bg='#363636')

        window_width = master.winfo_screenwidth()
        window_height = 800
        self.master.geometry(f"{window_width}x{window_height}+0+0")

        self.master.resizable(False, False)

        self.algorithm_label = tk.Label(self.master, text="Chọn thuật toán phát hiện biên", font=("Segoe UI", 20),
                                        fg='white', bg='#363636')
        self.algorithm_label.pack(pady=10)

        self.button = tk.PhotoImage(file="img/red_button.png")
        self.back_button = tk.Button(self.master, text="Trở về", command=self.go_back_home, image=self.button,
                                     fg='white', bg='#363636', compound='center', font=("Segoe UI", 15),
                                     cursor='hand2', activeforeground='white', activebackground='#363636',
                                     relief=tk.GROOVE, borderwidth=0)
        self.back_button.place(x=10, y=30)

        self.algorithm_combobox = ttk.Combobox(self.master, values=["1D", "Roberts", "Prewitt", "Sobel",
                                                                    "Laplacian", "Canny thủ công", "Canny cv2"],
                                               font=("Segoe UI", 13))
        self.algorithm_combobox.option_add('*TCombobox*Listbox.font', ("Segoe UI", 13))
        self.algorithm_combobox.pack(pady=0)

        # Create Frame
        self.left_frame = tk.Frame(self.master, width=window_width // 2, height=window_height, bg='#363636')
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.master, width=window_width // 2, height=window_height, bg='#363636')
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.big_button = tk.PhotoImage(file="img/big_button.png")
        self.input_button = tk.Button(self.left_frame, text="Chọn ảnh đầu vào", command=self.select_input_image,
                                      image=self.big_button, fg='white', bg='#363636', compound='center',
                                      font=("Segoe UI", 15), cursor='hand2', activeforeground='white',
                                      activebackground='#363636', relief=tk.GROOVE, borderwidth=0)
        self.input_button.pack(pady=20)

        self.output_button = tk.Button(self.right_frame, text="Hiển thị ảnh đầu ra", command=self.show_output_image,
                                       image=self.big_button, fg='white', bg='#363636', compound='center',
                                       font=("Segoe UI", 15), cursor='hand2', activeforeground='white',
                                       activebackground='#363636', relief=tk.GROOVE, borderwidth=0)
        self.output_button.pack(pady=20)

        self.input_image_path = ""
        self.output_image_path = ""

        # Create Canvas
        self.input_canvas = tk.Canvas(self.left_frame, bg="white", width=window_width // 2, height=window_height // 2)
        self.input_canvas.pack()
        self.output_canvas = tk.Canvas(self.right_frame, bg="white", width=window_width // 2, height=window_height // 2)
        self.output_canvas.pack()

    def go_back_home(self):
        self.master.destroy()
        self.home_window.deiconify()

    def select_input_image(self):
        file_path = filedialog.askopenfilename(title="Chọn ảnh đầu vào", filetypes=[("Image files", "*.png;*.jpg;"
                                                                                                    "*.jpeg")])
        if file_path:
            self.input_image_path = file_path
            print(f"Ảnh đầu vào đã chọn: {self.input_image_path}")

            # Show image
            self.show_input_image()

    def show_input_image(self):
        if self.input_image_path:
            # Show image
            input_image = Image.open(self.input_image_path)
            input_image = self.resize_image(input_image, self.input_canvas.winfo_width(),
                                            self.input_canvas.winfo_height())
            self.input_photo = ImageTk.PhotoImage(input_image)
            self.input_canvas.create_image(0, 0, anchor="nw", image=self.input_photo)

    def show_output_image(self):
        if self.input_image_path:
            edge_detector = EdgeDetection()
            output_image = None
            if self.algorithm_combobox.get() == "":
                messagebox.showwarning("Lỗi", "Vui lòng chọn thuật toán phát hiện biên trước.")
            elif self.algorithm_combobox.get() == "1D":
                output_image = edge_detector.apply_1d_operator(self.input_image_path)
            elif self.algorithm_combobox.get() == "Roberts":
                output_image = edge_detector.apply_roberts_operator(self.input_image_path)
            elif self.algorithm_combobox.get() == "Prewitt":
                output_image = edge_detector.apply_prewitt_operator(self.input_image_path)
            elif self.algorithm_combobox.get() == "Sobel":
                output_image = edge_detector.apply_sobel_operator(self.input_image_path)
            elif self.algorithm_combobox.get() == "Laplacian":
                output_image = edge_detector.apply_laplacian_operator(self.input_image_path)
            elif self.algorithm_combobox.get() == "Canny thủ công":
                output_image = edge_detector.apply_canny_operator1(self.input_image_path)
            elif self.algorithm_combobox.get() == "Canny cv2":
                output_image = edge_detector.apply_canny_operator2(self.input_image_path)

            # Save image
            self.output_image_path = "output_image.png"
            output_image.save(self.output_image_path)

            # Show image
            output_image = Image.open(self.output_image_path)
            output_image = self.resize_image(output_image, self.output_canvas.winfo_width(),
                                             self.output_canvas.winfo_height())
            self.output_photo = ImageTk.PhotoImage(output_image)
            self.output_canvas.create_image(0, 0, anchor="nw", image=self.output_photo)

        else:
            messagebox.showwarning("Lỗi", "Vui lòng chọn ảnh đầu vào trước.")

    @staticmethod
    def resize_image(image, width, height):
        if image.width > width or image.height > height:
            return image.resize((width, height), Image.LANCZOS)
        else:
            return image.resize((image.width, image.height), Image.LANCZOS)


class SegmentationPage:
    def __init__(self, master, home_window):
        self.master = master
        self.home_window = home_window
        home_window.withdraw()

        self.master.title("Phân vùng ảnh")
        self.master.configure(bg='#363636')

        window_width = master.winfo_screenwidth()
        window_height = 800
        self.master.geometry(f"{window_width}x{window_height}+0+0")

        self.master.resizable(False, False)

        self.algorithm_label = tk.Label(self.master, text="Chọn thuật toán phân vùng ảnh", font=("Segoe UI", 20),
                                        fg='white', bg='#363636')
        self.algorithm_label.pack(pady=10)

        self.button = tk.PhotoImage(file="img/red_button.png")
        self.back_button = tk.Button(self.master, text="Trở về", command=self.go_back_home, image=self.button,
                                     fg='white', bg='#363636', compound='center', font=("Segoe UI", 15),
                                     cursor='hand2', activeforeground='white', activebackground='#363636',
                                     relief=tk.GROOVE, borderwidth=0)
        self.back_button.place(x=10, y=30)

        # Add your segmentation algorithm options here
        self.algorithm_combobox = ttk.Combobox(self.master, values=["Otsu", "Đẳng liệu", "Đối xứng nền",
                                                                    "Tam giác"], font=("Segoe UI", 13))
        self.algorithm_combobox.option_add('*TCombobox*Listbox.font', ("Segoe UI", 13))
        self.algorithm_combobox.pack(pady=0)

        # Create Frame
        self.left_frame = tk.Frame(self.master, width=window_width // 2, height=window_height, bg='#363636')
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.master, width=window_width // 2, height=window_height, bg='#363636')
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.big_button = tk.PhotoImage(file="img/big_button.png")
        self.input_button = tk.Button(self.left_frame, text="Chọn ảnh đầu vào", command=self.select_input_image,
                                      image=self.big_button, fg='white', bg='#363636', compound='center',
                                      font=("Segoe UI", 15), cursor='hand2', activeforeground='white',
                                      activebackground='#363636', relief=tk.GROOVE, borderwidth=0)
        self.input_button.pack(pady=20)

        self.output_button = tk.Button(self.right_frame, text="Hiển thị ảnh đầu ra", command=self.show_output_image,
                                       image=self.big_button, fg='white', bg='#363636', compound='center',
                                       font=("Segoe UI", 15), cursor='hand2', activeforeground='white',
                                       activebackground='#363636', relief=tk.GROOVE, borderwidth=0)
        self.output_button.pack(pady=20)

        self.input_image_path = ""
        self.output_image_path = ""

        # Create Canvas
        self.input_canvas = tk.Canvas(self.left_frame, bg="white", width=window_width // 2, height=window_height // 2)
        self.input_canvas.pack()
        self.output_canvas = tk.Canvas(self.right_frame, bg="white", width=window_width // 2, height=window_height // 2)
        self.output_canvas.pack()

    def go_back_home(self):
        self.master.destroy()
        self.home_window.deiconify()

    def select_input_image(self):
        file_path = filedialog.askopenfilename(title="Chọn ảnh đầu vào", filetypes=[("Image files", "*.png;*.jpg;"
                                                                                                    "*.jpeg")])
        if file_path:
            self.input_image_path = file_path
            print(f"Ảnh đầu vào đã chọn: {self.input_image_path}")

            # Show image
            self.show_input_image()

    def show_input_image(self):
        if self.input_image_path:
            # Show image
            input_image = Image.open(self.input_image_path)
            input_image = self.resize_image(input_image, self.input_canvas.winfo_width(),
                                            self.input_canvas.winfo_height())
            self.input_photo = ImageTk.PhotoImage(input_image)
            self.input_canvas.create_image(0, 0, anchor="nw", image=self.input_photo)

    def show_output_image(self):
        if self.input_image_path:
            segmentation = Segmentation()
            output_image = None
            if self.algorithm_combobox.get() == "":
                messagebox.showwarning("Lỗi", "Vui lòng chọn thuật toán phân vùng ảnh trước.")
            elif self.algorithm_combobox.get() == "Otsu":
                output_image = segmentation.apply_otsu_segmentation(self.input_image_path)
            elif self.algorithm_combobox.get() == "Đẳng liệu":
                output_image = segmentation.apply_isodata_segmentation(self.input_image_path)
            elif self.algorithm_combobox.get() == "Đối xứng nền":
                output_image = segmentation.apply_background_symmetry_algorithm(self.input_image_path)
            elif self.algorithm_combobox.get() == "Tam giác":
                output_image = segmentation.apply_triangle_algorithm(self.input_image_path)

            # Save image
            self.output_image_path = "output_image.png"
            output_image.save(self.output_image_path)

            # Show image
            output_image = Image.open(self.output_image_path)
            output_image = self.resize_image(output_image, self.output_canvas.winfo_width(),
                                             self.output_canvas.winfo_height())
            self.output_photo = ImageTk.PhotoImage(output_image)
            self.output_canvas.create_image(0, 0, anchor="nw", image=self.output_photo)

        else:
            messagebox.showwarning("Lỗi", "Vui lòng chọn ảnh đầu vào trước.")

    @staticmethod
    def resize_image(image, width, height):
        if image.width > width or image.height > height:
            return image.resize((width, height), Image.LANCZOS)
        else:
            return image.resize((image.width, image.height), Image.LANCZOS)


class MorphologicalProcessingPage:
    def __init__(self, master, home_window):
        self.master = master
        self.home_window = home_window
        home_window.withdraw()

        self.master.title("Xử lý hình thái")
        self.master.configure(bg='#363636')

        window_width = master.winfo_screenwidth()
        window_height = 800
        self.master.geometry(f"{window_width}x{window_height}+0+0")

        self.master.resizable(False, False)

        self.algorithm_label = tk.Label(self.master, text="Chọn thuật toán xử lý hình thái", font=("Segoe UI", 20),
                                        fg='white', bg='#363636')
        self.algorithm_label.pack(pady=10)

        self.button = tk.PhotoImage(file="img/red_button.png")
        self.back_button = tk.Button(self.master, text="Trở về", command=self.go_back_home, image=self.button,
                                     fg='white', bg='#363636', compound='center', font=("Segoe UI", 15),
                                     cursor='hand2', activeforeground='white', activebackground='#363636',
                                     relief=tk.GROOVE, borderwidth=0)
        self.back_button.place(x=10, y=30)

        # Add your morphological processing algorithm options here
        self.algorithm_combobox = ttk.Combobox(self.master, values=["Phép co", "Phép giãn", "Phép mở", "Phép đóng"],
                                               font=("Segoe UI", 13))
        self.algorithm_combobox.option_add('*TCombobox*Listbox.font', ("Segoe UI", 13))
        self.algorithm_combobox.pack(pady=0)

        # Create Frame
        self.left_frame = tk.Frame(self.master, width=window_width // 2, height=window_height, bg='#363636')
        self.left_frame.pack(side="left", fill="both", expand=True)

        self.right_frame = tk.Frame(self.master, width=window_width // 2, height=window_height, bg='#363636')
        self.right_frame.pack(side="right", fill="both", expand=True)

        self.big_button = tk.PhotoImage(file="img/big_button.png")
        self.input_button = tk.Button(self.left_frame, text="Chọn ảnh đầu vào", command=self.select_input_image,
                                      image=self.big_button, fg='white', bg='#363636', compound='center',
                                      font=("Segoe UI", 15), cursor='hand2', activeforeground='white',
                                      activebackground='#363636', relief=tk.GROOVE, borderwidth=0)
        self.input_button.pack(pady=20)

        self.output_button = tk.Button(self.right_frame, text="Hiển thị ảnh đầu ra", command=self.show_output_image,
                                       image=self.big_button, fg='white', bg='#363636', compound='center',
                                       font=("Segoe UI", 15), cursor='hand2', activeforeground='white',
                                       activebackground='#363636', relief=tk.GROOVE, borderwidth=0)
        self.output_button.pack(pady=20)

        self.input_image_path = ""
        self.output_image_path = ""

        # Create Canvas
        self.input_canvas = tk.Canvas(self.left_frame, bg="white", width=window_width // 2, height=window_height // 2)
        self.input_canvas.pack()
        self.output_canvas = tk.Canvas(self.right_frame, bg="white", width=window_width // 2, height=window_height // 2)
        self.output_canvas.pack()

        self.kernel_size_label = tk.Label(self.right_frame, text="Kích thước bộ lọc", font=("Segoe UI", 12), fg='white',
                                          bg='#363636')
        self.kernel_size_label.pack(pady=2)
        self.kernel_size_entry = tk.Entry(self.right_frame, font=("Segoe UI", 12))
        self.kernel_size_entry.pack(pady=2)

    def go_back_home(self):
        self.master.destroy()
        self.home_window.deiconify()

    def select_input_image(self):
        file_path = filedialog.askopenfilename(title="Chọn ảnh đầu vào", filetypes=[("Image files", "*.png;*.jpg;"
                                                                                                    "*.jpeg")])
        if file_path:
            self.input_image_path = file_path
            print(f"Ảnh đầu vào đã chọn: {self.input_image_path}")

            # Show image
            self.show_input_image()

    def show_input_image(self):
        if self.input_image_path:
            # Show image
            input_image = Image.open(self.input_image_path)
            input_image = self.resize_image(input_image, self.input_canvas.winfo_width(),
                                            self.input_canvas.winfo_height())
            self.input_photo = ImageTk.PhotoImage(input_image)
            self.input_canvas.create_image(0, 0, anchor="nw", image=self.input_photo)

    def show_output_image(self):
        if self.input_image_path:
            morphological_processing = MorphologicalProcessing()
            output_image = None
            if self.algorithm_combobox.get() == "":
                messagebox.showwarning("Lỗi", "Vui lòng chọn thuật toán xử lý hình thái trước.")
            elif self.algorithm_combobox.get() == "Phép co":
                kernel_size_value = self.kernel_size_entry.get()
                if kernel_size_value == "":
                    messagebox.showwarning("Lỗi", "Vui lòng nhập kích thước bộ lọc.")
                else:
                    try:
                        output_image = morphological_processing.apply_erosion(self.input_image_path,
                                                                              kernel_size=int(kernel_size_value))
                    except ValueError:
                        messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")
            elif self.algorithm_combobox.get() == "Phép giãn":
                kernel_size_value = self.kernel_size_entry.get()
                if kernel_size_value == "":
                    messagebox.showwarning("Lỗi", "Vui lòng nhập kích thước bộ lọc.")
                else:
                    try:
                        output_image = morphological_processing.apply_dilation(self.input_image_path,
                                                                               kernel_size=int(kernel_size_value))
                    except ValueError:
                        messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")
            elif self.algorithm_combobox.get() == "Phép mở":
                kernel_size_value = self.kernel_size_entry.get()
                if kernel_size_value == "":
                    messagebox.showwarning("Lỗi", "Vui lòng nhập kích thước bộ lọc.")
                else:
                    try:
                        output_image = morphological_processing.apply_opening(self.input_image_path,
                                                                              kernel_size=int(kernel_size_value))
                    except ValueError:
                        messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")
            elif self.algorithm_combobox.get() == "Phép đóng":
                kernel_size_value = self.kernel_size_entry.get()
                if kernel_size_value == "":
                    messagebox.showwarning("Lỗi", "Vui lòng nhập kích thước bộ lọc.")
                else:
                    try:
                        output_image = morphological_processing.apply_closing(self.input_image_path,
                                                                              kernel_size=int(kernel_size_value))
                    except ValueError:
                        messagebox.showwarning("Lỗi", "Kích thước bộ lọc không hợp lệ.")

            # Save image
            self.output_image_path = "output_image.png"
            output_image.save(self.output_image_path)

            # Show image
            output_image = Image.open(self.output_image_path)
            output_image = self.resize_image(output_image, self.output_canvas.winfo_width(),
                                             self.output_canvas.winfo_height())
            self.output_photo = ImageTk.PhotoImage(output_image)
            self.output_canvas.create_image(0, 0, anchor="nw", image=self.output_photo)

        else:
            messagebox.showwarning("Lỗi", "Vui lòng chọn ảnh đầu vào trước.")

    @staticmethod
    def resize_image(image, width, height):
        if image.width > width or image.height > height:
            return image.resize((width, height), Image.LANCZOS)
        else:
            return image.resize((image.width, image.height), Image.LANCZOS)


if __name__ == "__main__":
    root = tk.Tk()
    app = HomePage(root)
    root.mainloop()

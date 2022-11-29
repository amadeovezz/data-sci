import numpy as np


def flip_kernel(kernel: np.ndarray) -> np.ndarray:
    """
    Flips a kernel so that it can be used for convolution.

    @param kernel: The kernel to flip
    @return:
    """
    return np.flip(np.flip(kernel, axis=0),1)


def filter(img: np.ndarray, kernel: np.ndarray, convolve: bool=False) -> np.ndarray:
    """
    @param img: The img to filter.
    @param kernel: The kernel to use on the image.
    @param convolve: If not correlation flip kernel for convolution

    Apply a kernel to an 2-d image. To handle boundary cases we
    create larger image with zeros.

    @return: The convolved image
    """

    if convolve:
        kernel = flip_kernel(kernel)

    # Infer kernel size (k value) recall kernels are 2k + 1 by 2k + 1
    # Thus take inverse of y = 2k + 1
    kernel_num_of_rows, kernel_num_of_columns = kernel.shape
    k = int((kernel_num_of_rows - 1) / 2)

    # Create new image for computation so that our kernel can 'move' across edges
    row_num, col_num = img.shape
    safe_img = np.zeros((row_num+(2*k),col_num+(2*k)), dtype=np.float)
    # Copy old image into new img
    safe_img[k:row_num+k, k:col_num+k] = img

    # Create a result image
    result_img = np.zeros(safe_img.shape, dtype=np.float)

    # Iterate through each element of our safe_img - starting at the beginning indexes
    # of our original image
    for i in range(k, row_num+k):
        for j in range(k, col_num+k):
            # Find the values in our safe img to multiply with our kernel
            # lower and upper bounds indexing is exclusive hence the + 1
            neighbourhood_values = safe_img[i-k:i+k+1, j-k:j+k+1]
            # Compute the Hadamard product and then sum matrix
            result_img[i,j] = np.multiply(neighbourhood_values, kernel).sum()

    return result_img[k:row_num+k, k:col_num+k]


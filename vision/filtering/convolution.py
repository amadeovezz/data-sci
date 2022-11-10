import numpy as np


def convolve(img: np.array, kernel: np.array) -> np.array:
    """
    @param img: The img to convolve.
    @param kernel: The kernel to use on the image.

    Apply a kernel to an 2-d image. To handle kernels on the edges
    a new larger image is created so that kernels can 'move' across
    the image without running into indexing errors.

    @return: The convolved image
    """

    # Infer kernel size (k value) recall kernels are 2k + 1 by 2k + 1
    # Thus take inverse of y = 2k + 1
    kernel_num_of_rows, kernel_num_of_columns = kernel.shape
    k = int((kernel_num_of_rows - 1) / 2)

    # Create new image for computation so that our kernel can 'move' across edges
    row_num, col_num = img.shape
    idx_img = np.zeros((row_num+(2*k),col_num+(2*k)), dtype=float)
    # Copy old image into new img
    idx_img[k:row_num+k, k:col_num+k] = img

    # Create a result image
    result_img = np.zeros(idx_img.shape, dtype=float)

    # Iterate through each element of the idx_img
    for i in range(k, row_num+k):
        for j in range(k, col_num+k):
            # Find the neighbourhood indexes in terms of i,j,k
            # lower and upper bounds indexing is exclusive hence the + 1
            neighbourhood_values = idx_img[i-k:i+k+1, j-k:j+k+1]
            # Compute the Hadamard product and then sum matrix
            result_img[i,j] = np.multiply(neighbourhood_values, kernel).sum()

    return result_img[k:row_num+k, k:col_num+k]



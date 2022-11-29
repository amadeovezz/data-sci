# Vision

This directory contains:

- A collection of image-processing and computer vision algorithms
- A thin client, img_cli that uses these algorithms
- Some jupyter notebooks that demonstrate img_cli's usage (see `tutorials/img-cli/img_cli.ipynb`)

API:

```python

# Image reading 

img = img_cli.ImgCli("../images/crocodile-1.jpg")

# Display

img.show()

# Reduce noise

img.reduce_noise()

# Gradient image

img.gradient_image()



```



The main dependencies used are:

- `numpy`: for computation
- `pillow`: for reading and displaying images
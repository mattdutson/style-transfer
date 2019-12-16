## Conda Environment

To create the `style-gpu` environment, run:
```
conda create -f envs/style-gpu.yml
```

To create the `style` environment, run:
```
conda create -f envs/style.yml
```

## Command-Line Usage

After installing the requisite packages (see "Conda Environment"), the style transfer algorithm can be run by navigating to the project base directory and running
```
./transfer.py <content_img> <style_img>
```
Where `<content_img>` and `<style_img>` are the filenames of the content and style images, respectively. By default, the result is saved to `out.png`. This can be changed using the `-o` argument.

A complete list of options can be seen by running
```
./transfer.py -h
```

## Python API

The `transfer` Python function can be imported via
```python
from transfer import transfer
```

This function takes the same arguments as the command-line interface and has the signature
```python
def transfer(
        content_img,
        style_img,
        content_size=None,
        style_size=None,
        content_layer="block4_conv2",
        content_weight=1.0,
        style_weight=10.0,
        variation_weight=0.01,
        noise=0.0,
        n_iter=500):
```
`content_img` and `style_img` should be NumPy arrays of shape `(w, h, 3)`. The function returns a NumPy array whose shape matches that of `content_img` (after being downscaled according to `content_size`).

See the `playground.ipynb` Jupyter notebook for examples of Python API usage.

## Tips

The computational cost of the transfer is proportional to the number of pixels in the content image. I've found that, with a good GPU, the transfer takes about 5 minutes on a 512 by 512 image. I wouldn't recommend making the content image much larger than 1024 by 1024. The size of the content image can be changed by setting `content_size`.

The "brush size" (e.g. zoom level of the texture) can be controlled by setting the value of `style_size`. The absolute size (in pixels) of a texture matters!

The balance between matching the content image, matching the style image, and reducing noise can be tuned with the `content_weight`, `style_weight`, and `variation_weight` arguments, respectively. You might not notice a difference until you change a weights by a factor of ~100.

## Sample Images

The `content` and `style` subdirectories contain images for testing. Many of these are quite large, so it's very important you set `content_size` and `style_size` to reasonable values!

The style images (paintings) should all be public domain. The content images are all photos I've taken myself. Feel free to use them without restriction. 

## Attributions

This code is primarily based on [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576) (Gatys et al. 2015). [This blog post](http://www.subsubroutine.com/sub-subroutine/2016/11/12/painting-like-van-gogh-with-convolutional-neural-networks) was also a source of guidance. The total variation loss term comes from [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155) (Johnson et al. 2016).
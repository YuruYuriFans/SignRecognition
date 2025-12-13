# 1 Image Representation and Enhancement

digitization –  the process of converting a signal from analogue to digital form.

- Sampling, when we measure the signal’s value at discrete intervals;(horizontal)

- Quantization, when we restrict the value to a fixed set of levels.(vertical)

    - Sampling and quantization can be carried out in either order, by
special hardware devices – analogue to digital converters (ADCs).

sampling rate: The number of samples in a fixed amount of time or space.

The Shannon Sampling Theorems – if the highest frequency
component of a signal is at $f_h$ , the signal can be properly
reconstructed if it has been sampled at a frequency greater than $2f_h$.


- This limiting value is known as the Nyquist rate.

aliasing: a signal undersappled at **less than the Nyquist rate**, some frequency components in the original will get
transformed into other frequencies when the signal is reconstructed.
- in sound, heard as disortion
- in images, seen jagged edges, or  where the
image contains fine repeating details (e.g., Moire patterns)
- in moving pictures, jerkiness of
motion, as well as phenomena similar to the retrograde disk just described.

Human ear detecting range: 20Hz-20kHz. Therefore, If the limit of hearing is taken to be 20kHz, a minimum sampling rate
of 40 kHz is required by the Sampling Theorem.

- Audio CDs: 44.1kHZ

- Audio over internet: 22.05kHz 

- Speech: 11.024kHz

- DAT(dtigal audio tape): 48kHz, best quality desired.

Digital Data Acquisition

- physical device

- digitizer

three principal sensor
arrangements used to transform
illumination energy into digital images: Single imaging sensor, Line sensor, Array sensor.

- Single imaging sensor: In order to generate a 2D image using
a single sensor, there has to be
relative displacements in both the x-
and y-directions between the sensor
and the area to be imaged.

  - arrangement used in high-
precision scanning, inexpensive (but slow) way
to obtain high-resolution images. 

- Line sensor: A geometry that is used much more
frequently than single sensors, consists
of an in-line arrangement of sensors in
the form of a sensor strip.

    - The strip provides imaging elements in one
direction. Motion perpendicular to the strip
provides imaging in the other direction.

- The imaging strip gives **one line** of an image at
a time, and the motion of the strip completes
the other dimension of a 2D image.
    
  - This is the type of arrangement used in most
flat bed scanners.


- Array sensor: Individual sensors arranged in the form of a 2D array.

    - Numerous electromagnetic, some ultrasonic sensing devices, **predominant** arrangement in digital cameras.

The result of sampling and quantization is a **matrix** of real numbers.

**Visual perception**: a first step in the study of multimedia.
- Brightness adaptation and discrimination

discriminate between
**different intensity levels** is an important consideration in
presenting image-processing results
- Experimental evidence indicates that subjective brightness
(intensity as perceived by the human visual system) is **not**
a simple function of intensity.

**Phenomenon 1: Mach bands**

The human visual
system tends to undershoot or overshoot
around the boundary of regions of different
intensities.

**Phenomenon 2: simultaneous contrast**: they appear to the eye to become progressively darker as the background becomes lighter.


A more familiar example is a piece of paper that seems white when lying on a
desk, but can appear totally black when used to shield the eyes while looking
directly at a bright sky.

**Phenomenon 3: optical illusion**: The eye fills in non-existing
information or wrongly perceives geometrical properties of objects. Not fully understood yet. 

Application of Visual Perception : 

- Edge Detection, 

- Anti-aliasing:

- Motion blur:  When recording reality with a video or a film camera, we observe 
that objects that move too fast in front of the camera appear blurred. (where the shutter speed is too slow).

  - Motion blur can add a touch of realism to computer animation because
it reminds viewers of the blurring effect that occurs when we record
fast-moving real objects directly with a camera. But motion blur **does
not occur naturally in computer animation**, it must be added.

One of the common techniques / programs use to create motion blur is rendering
the scene a number of times while advancing the animation slightly. 

The multiple images are then composited together into a single, motion-blurred image.

However, this means a 4-5 fold increase in rendering time.

Image Compression: 

- false contouring:  almost imperceptible set of very fine ridge-like structures in areas of smooth gray levels

  - generally is quite visible in images displayed using 16 or less uniformly spaced gray levels.

Digital Image Representation: Each element of the matrix array is called an image element,
picture element, pixel, or pel.

$b = M*N*k$

When an image can have 2k gray levels, it is common practice to
refer to the image as a “k-bit image”.

Sampling is the principal factor determining the spatial resolution
of an image.

Basically, spatial resolution is the smallest discernible detail in an
image.

Digital Image Processing:  two broad categories: spatial domain methods and frequency domain methods.

- The spatial domain processing techniques are based on direct
manipulation of pixels in an image;

- The frequency domain processing techniques are based on
modifying the Fourier (or others) transform of an image.



Image enhancement: one of the most interesting and visually appealing
areas of image processing. Image features such as boundaries, edges, and contrast, etc, are enhanced
for display and analysis. The information of the image will not be increased but the chosen features.

- Contrast stretching

- Histogram equalization

- Noise smoothing 

-  filtering

The function or **parameter curves** are
graphs that represent and control different
attributes of an image, such as **brightness**
or **color**. 

- These attributes can be easily
modified by manipulating the function
curves without having to alter the image
directly with a retouching tool.

  - Making
image manipulations that involve all of the
image or large portions of it are best
performed with function curves.


- Function curves for image manipulation are
usually represented by a line that starts at the
lower left corner of a square and ends at the
upper right corner. 

  - The straight diagonal line
represents one or several untouched attributes
of the original image. 

    - Any changes made to the line will result in changes to the image.



Three basic types of GLT ( Gray Level Transformations) functions
used for image enhancement:

- Linear (negative and identity);

- Logarithmic (log and inverse-log);

- Power-law (nth power and n-th root)

Linear, negative: 

Log transformation: $s = c log(1+|r|)$

- sometimes the dynamic range of an image exceeds the capability of the display
device: only the brightest parts of the image are visible on the display.

- can map a narrow range of low gray-level values in the input image into a wider
range of output levels. 
  - The opposite is true of higher values of input levels.

- allow to expand the values of dark pixels in an image while compressing the
higher-level values.

Power-Law Transformation: $s = c r^{\gamma}$, where c and $\gamma$ are positive constants.

- Power-law curves with
fractional values of γ map a
narrow range of dark input
values into a wider range of
output values, 
  - with the
opposite being true for higher
values of input levels.

- Unlike the log function, a
family of possible
transformation curves
obtained simply by varying γ.

gamma > 1:  compressing the low end and expanding the high end of the gray scale.


Piecewise-Linear Transformation: A complementary approach to the previous three basic transformations
is to use piecewise linear functions.

- Advantage: It can be arbitrarily complex.

- Disadvantage: It requires considerably more user input.

- Two typical functions: contrast-stretching and gray-level slicing.

Contrast Stretching: 

- Why:  Low-contrast images can result
from poor illumination, lack of
dynamic range in the imaging
sensor, or even wrong setting of
a lens aperture during image
acquisition.

- Idea: increase the
dynamic range of the gray
levels 

> Dynamic range refers to the ratio between the loudest and quietest signals a system (like a camera, microphone, or audio system) can handle while maintaining accuracy. It essentially defines the range of detectable values within a signal, from the very faint to the very strong.


Window-Level Operation (a.k.a. Window-Center Adjustment): In this operation, an interval or window is selected in the original gray level
range, determined by the window center or level l, and the window width w. Explicitly

- Contrast outside the window is lost completely, whereas the portion of the range lying inside
the window is stretched to the original gray level range.

Such GLT is useful for highlighting an intensity band of interest.

Gray-level slicing: To highlight a specific range of gray levels in an image, such as
enhancing flaws in X-ray images.

- Approach 1
Highlight range [A, B]
of gray levels and
reduce all others to a
constant level

- Approach 2
Highlight range [A, B]
but preserve all other
levels

#### Histogram processing

histogram of a digital image with gray levels in the range [0, L-1] is
a discrete function: 

$$h(r_k) = n_k$$

where rk is the k-th gray level and n k is the number of pixels in the image
having gray level rk .

A histogram provides a view of the intensity profile of an image and is
often displayed as a bar chart.

 Pixel values are partitioned and counted with the population of each
partition value placed in its own bin.

The pixel intensities are plotted along the horizontal x-axis while the
number of occurrences for each intensity are plotted along the vertical
y-axis.

Histograms can be viewed as probability density functions.

Normalized Histogram:  dividing each of its
values by the total number of pixels in the image, denoted by n.

$$p(r_k) = n_k / n$$

An image whose pixels tend to occupy the **entire range** of possible
gray levels and tend to be **distributed uniformly**, will have an
appearance of **high contrast** and will exhibit a large variety of gray
tones. 

It is possible to develop a transformation function that can
automatically achieve the above effect, based only on information
available in the histogram of the input image.

- The appearance of the histogram of an image gives useful
information for possible contrast enhancement.

Histogram equalization: A mapping to increase the contrast in an image by stretching its
histogram to approximately uniformly distributed

- The image that has been histogram equalized always has pixels that
reach the brightest grey level

Histogram equalization requires a mapping (via a transformation) to
stretch the histogram of the input image

$$s_k = T(r_k) = \sum^{k}_{j=0} \frac{n_j}{n}$$

- rk is the k-th (k = 0, 1, …, L-1) grey level

- nk is the number of pixels in the image with that grey level

- n is the number of pixels in the image

- s has as many elements as the original histogram

#### Spatial Filtering

Spatial domain processes can be denoted by the expression: $g(x,y) = T[f(x,y)]$

- f(x, y) input image

- g(x, y) processed image

- T is an operator on f, defined over some neighborhood of (x.y)
- 
The sub-image is called a (spatial) filter, mask, kernel, template, or window.

The values in a filter sub-image are referred to as **coefficients**, rather than pixels.

The above process of linear filtering is similar to a frequency domain concept
called *convolution*, and is often referred to as “convolving a mask with an image”.
Similarly, filter masks are sometimes called *convolution masks*.

Smoothing filters are used for blurring and for noise reduction.

-  Linear smoothing filters(aka averaging filters) (e.g. box filter, weighted filter)

-  Non-linear smoothing filters (e.g. median filter, max filter, min filter)

An important application of spatial averaging is to blur an image for the
purpose getting a gross representation of **objects of interest**, 

- such that the intensity of smaller object blends with the background 

- and larger objects become “blob-like” and easy to detect.

Order-statistics smoothing filters are non-linear spatial filters whose
response is based on ordering (ranking) the pixels contained in the
image area encompassed by the filter, 

- and then replacing the value of the center pixel with the value determined by the ranking result.

The best-known example is the median filter

- which replaces the value of a pixel by the median of the gray levels in the neighborhood of that
pixel (the original value of the pixel is included in the computation of the
median).

The median, ξ, of a set of values is such that half the values in the set
are less than or equal to ξ, and half are greater than or equal to ξ.

In order to perform median filtering at a point in an image, we first **sort**
the values of the pixel and its neighbors, **determine** their median, and
**assign** this value to that pixel.

The principal function of median filters is to force points with distinct gray
levels to be more like their neighbors.

-  In fact, isolated clusters of pixels that are light or dark with respect to their
neighbors, and whose area is less than n 2 / 2 (one-half the filter area), are
eliminated by an n x n median filter. In this case “eliminated” means forced
to the median intensity of the neighbors. Larger clusters are affected
considerably less.

For certain types of random noise, median filters provide excellent
noise-reduction capabilities, with considerably less blurring than linear
smoothing filters of similar size.

- Median filters are particularly effective in the presence of **impulse noise**,
also called **salt-and-pepper noise** because of its appearance as **white and
black dots** superimposed on an image.

Image sharpening: principal objective of sharpening is to **highlight fine detail** in an image
or **enhance detail** that has been **blurred**, either in error or as a natural
effect of a particular method of image acquisition.

The Unsharp Masking approach* (The name “Unsharp Masking” is misleading because this filter actually sharpens images. The name is derived from the way the mask is constructed.)

- A process that has been used for many years by the printing and publishing
industry to sharpen images consists of subtracting an unsharp (smoothed)
version of an image from the original image.

- Variations of the unsharp mask can be created using different sizes of masks
or types of blur / smoothed masks.

A typical image
sharpening filter mask
used to implement the
linear **Laplacian operator**

# 2 Morphological Processing

Mathematical morphology is very often used in applications where
**shape** of objects and **speed** are major issues.


Applications: analysis of microscopic images (in biology, material
science, geology, and criminology), industrial inspection, optical
character recognition, and document analysis etc.

Digital image:

- (x,y) are integers from $Z^2$

- and $f$ is mapping that assigns an intensity value 

  -  (that is, a real number from the set of real numbers, R)
to each distinct pair of coordinates (x, y).

- If the elements of R also are integers, a digital image then becomes a
two-dimensional function whose coordinates and amplitude (i.e.,
intensity) values are integers.

Two sets A and B are said to be disjoint or mutually exclusive if they
have no common elements.
  - $A \cap B = \emptyset$

**complement** of a set A is the set of elements not contained in A:
- $A^c = \{ w | w \notin A \}$

difference of two sets A and B, denoted A – B, is defined as:
- $A - B = \{ w | w \in A, w \notin B \} = A \cap B^c$
  - We see that this is the set of elements that belong to A, but not to B.

Two additional definitions that are used extensively in morphology but generally are not found in basic texts on set theory.

**Translation** of set A by point z = (z1, z2), denoted $(A)_z$ , is defined as
- $(A)_z = \{ c | c=a+z, \text{for } a \in A \}$ 

**Reflection** of set B, denoted B̂ , is defined as
- $\hat{B} = \{w|w=-b,\text{for }b \in B\}$

## Primitive morphological operations, Dilation and Erosion

**Dilation** is an operation that “grows” or “thickens” objects in a binary
image. The specific manner and extent of this thickening is controlled
by a shape referred to as a ***structuring element***.

> Dilation expands the connected sets of the 1s of a binary image.
- > Growing features, filling holes and gaps

**Erosion** “shrinks” or “thins” objects in a binary image. The manner
and extent of shrinking is also controlled by a *structuring element*.

> Erosian shrinks the connected sets of 1s of a binary image
- > shrinking features, removing bridges, branches, rotrusions.

### Dilation

$$ A \oplus B = \{ z \mid (\hat{B})_z \cap A \neq \emptyset \} $$
   * Here, $\hat{B}$ is the reflection of the structuring element $B$ about the origin.
   * $(\hat{B})_z$ means we translate this reflection by $z$.
   * The condition $(\hat{B})_z \cap A \neq \emptyset$ means **at least one pixel of the shifted structuring element overlaps with $A$**.
   * So dilation collects all displacements $z$ where there is some overlap.

$$ A \oplus B = \{ z \mid ((\hat{B})_z \cap A) \subseteq A \} $$
   * This is a reformulation: since any non-empty overlap with $A$ is enough to make $z$ part of the dilation, the condition can be described as the set of $z$ for which the overlap region $(\hat{B})_z \cap A$ consists of elements of $A$ (which is always true if the intersection is non-empty).
   * Put another way: requiring $(\hat{B})_z \cap A \subseteq A$ ensures that we only pick $z$ where the overlap contributes points of $A$ (not the background).

> Set B is commonly referred to as the structuring element in dilation, as
well as in other morphological operations.

Why they are equivalent

* The **first definition** emphasizes the **existence of overlap**.
* The **second definition** emphasizes that the overlap is indeed **a subset of $A$** (which is tautologically true once overlap occurs).
 - Thus, both capture the same idea: dilation adds all positions $z$ where the shifted structuring element touches $A$.


### Erosion
$$ A \ominus B = \{ z \mid B_z \subseteq A \} $$

* Here $B_z$ is the translation of structuring element $B$ by $z$.
* The condition means: keep all points $z$ such that **when $B$ is placed at $z$, it fits entirely inside $A$**.
* So erosion “shrinks” objects, removing boundary pixels where the structuring element does not fit.


$$ A \ominus B = \{ z \mid (B)_z \cap A^c = \emptyset \} $$

* Equivalent formulation: erosion is the set of positions $z$ where the translated $B$ **does not overlap with the background** ($A^c$).
* Intuitively: the structuring element must be fully inside the object, otherwise some part would touch the background.


Why they are equivalent:
* Saying **$B_z \subseteq A$** means every element of $B_z$ is inside $A$.
* Equivalently, this means **$B_z$ has no elements in $A^c$**, i.e.,

  $$
  B_z \cap A^c = \emptyset
  $$
* Therefore, both definitions describe the **same operation**: erosion collects all points where $B$, shifted to that location, lies completely within $A$.


## Opening / Closing.

Dilation and erosion are not inverse. 
- If eroded and dilated, not the same. Instead, it is simplified and less-detailed. 

Opening: Erosion followed by dilation
- generally smooths the contour of an object, breaks narrow
isthmuses, and eliminates thin protrusions.


Closing: Dilation followed by erosion
- also tends to smooth sections of contours 
  - but, as opposed to opening, it generally fuses narrow breaks and long thin gulfs,
eliminates small holes, and fills gaps in the contour.

Opening and closing are useful when thresholding
(or some other initial process) produces a binary image with tiny holes in the
connected components or with a pair of components that should be separate
joined by a thin region of foreground pixels.

Opening: $A \circ B = (A \ominus B) \oplus B$

Closing: $A \bullet B = (A \oplus B) \ominus B$

## Morphological Filters
In practical image processing applications, dilation, erosion, opening,
and closing are used most often in various combinations.

Morphological operations can be used to construct morphological filters similar to
the spatial filters.

> Spatial filters: Operate on images by modifying pixel values using their neighbors, typically through convolution or correlation with a kernel (mask).

> Morphological Filters: Operate on images based on the shape and structure of objects, using set theory rather than pixel intensities.


Boundary Extraction: $\beta(A) = A - (A \ominus B)$, where B is a suitable structuring element.
- The boundary of A, denoted by $\beta(A)$, can be obtained by first
eroding A by B and then performing the set difference between A and
its erosion.

# 3 Color Models \& Color Image Processing

Humans can discern thousands of color shades and intensities, compared to about
only two dozen shades of gray.

the wavelength of visible light lies roughly between 400nm
and 700nm – and its intensity.

We can combine these measurements into a **spectral power distribution**
(SPD), a concentration function of wavelength, to describe how the
intensity of light from some particular source varies with wavelength.
- However, SPDs are too cumbersome to work with when we are specifying
colors for use in computer systems, so we need to adopt a different approach.

*tristimulus theory* – any color can be
specified by just three values, giving the weights of each of three components.

We call red, green and blue the **additive primary colors**.

A **color model** (also called **color space** or **color system**)
is a specification of a coordinate system and a subspace
within that system where each color is represented by a
single point.

- Most color models in use today are oriented either
toward 
  - hardware (e.g., color monitors and printers) 
  - or toward applications where color manipulation is a goal
(e.g., in the creation of color graphics for animation).

## Color Representation and Color Models

RGB model
- most important, because it corresponds to the way in which
color is produced on computer color **monitors**, and it is also how color is
detected by **scanners**.

CMYK Model: better for color printing

HSV(also called HSB or HSI): Hue, Satuation, Value
- corresponds closely with the way humans describe and interpret color

### RGB
The color subspace of interest is the cube, in which RGB values are at three
corners; cyan, magenta, and yellow are at three other corners; black is at the
origin; and white is at the corner farthest from the origin.

For convenience, the assumption is that all
color values have been normalized so that the
cube is the unit cube, i.e., all values of R, G,
and B are assumed to be in the range [0,1].

In this model, the gray scale (points of equal
RGB values) extends from black to white
along the line joining these two points.

256 is a very convenient number to use in a digital representation,
since a single 8-bit byte can hold exactly that many different
values, usually considered as numbers in the range 0 to 255. Thus,
an RGB color can be represented in three bytes, or 24 bits.

The number of bits used to hold a color value is often referred to as
the **color depth**.
- common color depths are sometimes distinguished by the
terms millions of colors (24 bit), thousands of colors (16 bit) and
256 colors (8 bit).

### HSV Color Model
Unfortunately, the RGB, CMYK and other similar color models are
not well suited for describing colors in terms that are practical for
human interpretation.

Variations on the HSV model include **HSI** and **HSB**, in which the **third component** is either Intensity or Brightness, respectively.

The **hue** of a pixel refers to its **basic color** – such as red or yellow or violet or
magenta. It is usually represented in the **range of 0 to 360**, referring to the color’s
location ( in degree ) around a circular color palette. 
- For example, the color located at 90° corresponds to a yellow green.

**Saturation** is the brilliance or purity of the specific hue that is present in the
pixel. If we look again at the HSV color wheel, 
- colors on the perimeter are fully saturated, 
- and the saturation decreases as you move to the center of the wheel.

**Value** can just be thought of as the brightness of the color, although strictly
speaking it is defined to be the maximum of red, green, or blue values. 
- Trying to
represent this third component means that we need to move beyond a 2D graph.
  - The value is graphed along the third axis, with the lowest value, black, being
located at the bottom of the cylinder. White, the highest brightness value, is
consequently located at the opposite end.

if we take the color cube and stand it on the
black (0,0,0) vertex, with the white vertex (1,1,1) directly above it. Then the intensity (gray
scale) is along the vertical line joining these two vertices. 

We also note with a little thought that
the saturation (purity) of a color increases as a function of distance from the intensity axis. In
fact, the saturation of points on the intensity axis is zero, as evidenced by the fact that all points
along this axis are gray.


#### **Color Manipulations – HSV manipulations**

Even though we have talked about the HSV color space as an alternate method of
representing data, it is **generally not used to store** images. 
-  much more commonly employed for **manipulating** colors.
  - For example, affecting the saturation of an image merely by adjusting RGB values can
be rather cumbersome, but using the HSV model allows us to directly access the
saturation component, making manipulation trivial.

**Pseudocolor (false color) image processing** consists of **assigning
colors to gray values** based on a specified criterion.

##### Intensity Slicing Coding

##### Full-Color Image Processing
Two major categories:

**Category 1**: we process each component image individually and then
form a composite processed color image from the individually processed
components. (C1)
- histogram equalization


**Category 2**: we work with color pixels directly. Because color images
have at least three components, color pixels really are vectors. (C2)
- For an image of size M x N, there are MN
such vectors, c(x,y), for x = 0,1,2,…,M-1; y =
0,1,2,…,N-1.
- Spatial filtering (image sharpening)

We can provide a brief equation to describe the operator. We usually treat the entire image
as if it were a single variable. In these cases, we use the following conventions:

- I = Input image; 

- O = Output image. 

- Thus, O = I ´ 2.0 would refer to the above example, in
which every pixel in the input image was multiplied by 2.0 to produce the output image.

**RGB multiply**: Most of the color-correction operators can be applied either to all the channels of an image
equally or to individual channels in varying amounts. When applied equally, the result will
tend to be an overall brightness or contrast modification. When different amounts are applied
to different channels, a visual color shift will usually take place as well.

**Add**: Instead of affecting the apparent brightness of an image by multiplying, we can add (or
subtract) a constant value from each pixel. 
- Notice that, unlike the multiplication operation (which keeps the deepest blacks at the same level), the blacks in this example have gone to
gray. ( O = I + 0.2 )

**Gamma Correction**:  uses an exponential function $O = I^\frac{1}{\gamma}$. In other words, we raise the value of each pixel to power of 1 divided by the gamma value supplied.

- The reason why the Gamma operator is so popular becomes
apparent when you examine what happens when you raise 0 to any
power – it stays at 0*. A similar thing happens when 1.0 is raised to
any power – it stays at 1.0. 
  - In other words, no matter what gamma correction you apply to an image, pixels with a value of 0 or 1.0 will remain unchanged. 
    - The only effect that the gamma operator has will be on non-black and non-white pixels. It makes the image tend to look more natural.

**Invert**: An extremely simple operator: $O = (1 – I)$. Every pixel is replaced by the value of that pixel
subtracted from 1.0. The result is an image that appears to be similar to the photographic
negative of the original.

**Contrast**: a combination of the “Multiply” and “Subtract”. such as: $O = ( I – 0.33 ) * 3$.
- However, contrast as applied in this manner is a less than ideal operator, since both the low
end and the high end treat the image’s data rather harshly.
  - A better color manipulation is to apply gamma-like curves to the upper and lower ranges.
This method tends to give a much cleaner image, particularly at the low and high ends of the
brightness spectrum.

LUT Manipulations: An **input-to-output mapping** is known as a **look-up table (LUT)**. Such LUT-manipulation can
give an extremely fine amount of control, even allowing specific color corrections across a
narrow range of values. Here the output image has had all three channels modified by the
set of three user-defined curves.

## Image Restoration and Retouching
Image enhancement is largely a **subjective** process,  image restoration and retouching are for the most part an **objective** process, such as geometric transformation.
- Geometric transformations modify the spatial relationships between
pixels in an image. Such transform operation causes some or all of
the pixels in a given image to change their existing location.
  - often are called *rubber-sheet transformations*
    - because they may be viewed as the process of
“printing” an image on a sheet of rubber and then **stretching** this sheet
according to some predefined set of rules.

**Geometric Transformations** can be used for retouching images to
obtain effects that would otherwise be impossible..
- Such effects include panning, rotating, scaling, warping, and various
specialized distortion effects.

A **working resolution** is typically the resolution of the image that will be
produced once we are finished with our compositing operations.

**Panning**: If you wish to apply a simple translation to the image, by offsetting it in
both X and Y. Such a translation is usually referred to as a pan.
- On most settings, the out-of-border rest of the
image will be cropped, or discarded. Some settings can allow the off-screen information
to be preserved so that it can later be brought back into frame if needed.

**Rotation**: 

**Scale**:

**Warping**:Conceptually, it is easiest to think of warping as if your image were printed on a thin sheet of flexible rubber.
- This rubber sheet can be pushed and pulled by various amounts in various areas until the
desired result is obtained. 
  - Image warping is usually controlled by either a grid mesh or a series of splines.

### Image Compositing

Image Compositing consists of combining two or more different
images into one in such a way that an *illusion of time and space* is created. It seems that all the images happened at the **same time and
place** and were recorded together.

> One of the main purposes of image compositing is usually to save
expensive production costs or to simulate something that is physically
impossible to create in our reality.

The process of compositing images from different sources into a
single visually coherent image can be performed on **both still and
moving** images. Still composites are often called **collages**, while
moving composites result in dynamic composites or transition effects.

#### Image Compositing

#### multisource operators

**ADD**:

**Mattes** are used during compositing when we only wish a portion
of a certain image to be included in the result.

Mattes are generally considered to be single-channel, grayscale
images. There is no need for three separate channels, as there is
when specifying color, since the transparency for any given pixel
can be described by a single numerical value in the range of 0 to 1.

**OVER**:  takes two images and, using a third image
as a controlling matte, lays a portion of the first image on top of
the second.
- Mathematically, when we place image A (the foreground) over
image B (the background), using image M as the matte for
image A, our output image O is as follows:
  - $O = (A*M)+((1-M)*B)$

**MIX**: the weighted, normalized addition of two images. In other
words, the two images are averaged together, often with one of the
images contributing a larger percentage to the output.
- The equation for such a mix, where **“MV”** refers to the **mix value**
(the percentage of the first image that we will be using), is as
follows:
  - $O = (MV*A) + ((1-MV)*B)$


**Masks**:There are times when we wish to **limit the extent of a certain operator’s effect**.
A particularly useful method for doing this involves the use of a **separate matte** (aka **mask**)as a control image.

# W4

Two ways:

- video 

- animation


A video image is a projection of a 3D scene onto a 2D plane.

Video Color Systems: Largely derive from older analog methods of coding color for TV. Luminance is separated from color information.

> In Y.C.C. scheme, the “Y” was the same old luminance (brightness)
signal that was used by black and white televisions, while the “C’s”
stood for the Color components.

The two color components would determine the hue of a pixel, while
the luminance signal would determine its brightness. Thus, color
transmission was facilitated while **black and white compatibility** was
maintained.

There are numerous video color systems :
- YIQ color system (NTSC)
- YUV color system (PAL)
- YDrDb color system (SECAM)

Digital video is the representation of a spatio-temporally sampled
video scene in digital form.

typical processing steps involved in the digitization of video: After signal acquisition and amplification, the key processing steps are spatial sampling, temporal sampling, and quantization.

#### Spatial sampling

Spatial sampling consists of taking measurements of the underlying analog
signal at a finite set of sampling points in a finite viewing area (or frame).
- To simplify the process, the sampling points are restricted to lie on a lattice,
usually a rectangular grid.


The two-dimensional set of sampling points are transformed into a
one-dimensional set through a process called **raster scanning**.

The two main ways to perform raster scanning are **progressive
scanning** and **interlaced scanning**.

In a progressive (or non-interlaced) scan, the sampling points are
scanned from left to right and top to bottom.

- Progressive scanning is typically used for film and computer displays.

In an interlaced scan, the points are divided into **odd and even** scan lines.
The odd lines are scanned first from left to right and top to bottom. Then the
even lines are scanned.
- The odd (respectively, even) scan lines make up a **field**. In an interlaced scan,
two fields make up a frame.
- Interlaced scanning is commonly used for television signals.

De-interlace Processing

Interlaced video sometimes may produce unpleasant visual artifacts when
displaying certain textures or types of motion.
- If you are capturing images from a video signal, you can filter them
through a de-interlacing filter provided by image-editing applications.

**Video image resolutions**: The visual quality of the video image is influenced by the number of sampling points.
More sampling points (a higher sampling resolution) give a “finer” representation of the
image: however, more sampling points require higher storage capacity.
Typical video image resolutions

#### Temporal sampling

16 frames/sec: illusion of motion

24 frames/sec: motion picture tech

Why Digital Video
- Direct access, which makes nonlinear video editing simple
- Repeated recording without degradation of image quality
- Ease of encryption and better tolerance to channel noise

Digital Video Standards
- CCIR–601 standard
- Source Input Format (SIF)
- Common Intermediate Format (CIF)
- Quarter–CIF (QCIF)

#### CCIR–601 Standard

Since the YIQ, YUV, and YD rDb color systems are designed for analog
television, these systems are inherently analog.

CCIR–601 defines a standard digital representation of video in terms of digital YCrCb color components.

CCIR-601 defines both 8-bit and 10-bit digital encodings.

- In the 8-bit encoding,
assuming that the RGB components have been digitized to the range [0, 255],


the human visual system’s reduced spatial sensitivity to color

With 4:2:2 chroma subsampling, the two chroma components are subsampled by a
factor of two horizontally.

## Video Segmentation

Partition a video stream into a set of meaningful and
manageable segments, which then serve as the basic unit
for indexing

Semi-automatic or automatic segmentation
- Camera shot transitions
  - Abrupt transitions occur when two individual shots are simply pasted
together;
  - Gradual transitions connect two shots smoothly by applying special
editing techniques, such as fade, dissolve.
- Segmentation cues: video frame data, audio data, closed caption.

### Shot Cut Detection

Intensity and Color Template Matching

Histogram-based Matching: Two frames with minor changes in their intensity/color distributions
are likely of a similar scene
- i = 1
A scene change is declared whenever $S_2 ( f_m , f_n )$ exceeds a pre-specified threshold

### Motion Estimation

Motion estimation is the estimation of the parameters of a video model
that describes the temporal variations, usually from consecutive frames.

Applications:
- Motion compensated prediction (video compression)
- Motion compensated interpolation (video stabilization)
- Video image sequence analysis (computer vision)

2D Motion Model

The human eye perceives motion by identifying corresponding points at
different times.

The correspondence is usually determined by **assuming** that the **color or
brightness of a point does not change** after the motion.

Observed or apparent 2D motion is referred to as optical flow in
computer vision literature.

- Case1: a sphere with a uniform flat
surface is rotating under a constant
ambient light, but the observed image
does not change.

- Case2: a point light source is rotating
around a stationary sphere, causing the
highlight point on the sphere to rotate.

# 6

bitmapped image: cannot, without loss of quality, be scaled.(.jpeg, .gif, .png)

vector graphics: can be scaled or re-sized without loss of
quality. (.svg)

the name “graphics” reserved for vector
graphics, and “images” used to mean bitmapped images.

The basic objective of computer graphics is to build / model a
virtual world of graphics objects and to render a scene of the
virtual model from a specific view onto a graphic device.

This common space to host all the graphics objects is
often called the world space. A rendered scene of the world space,
the main output of a graphics system, is typically in a 2D form.

## 2D

A 2D graphics system models the virtual world with a two-
dimensional space.

Compared to 3D graphics, 2D graphics is simpler in both
modeling and rendering. 2D objects are easier to create and
manipulate. The 2D rendering usually does not involve any
complicated projections such as those in 3D graphics.

Even though a 2D model cannot completely capture the full nature
of a 3D space, 2D computer graphics is widely applied because of
its simplicity and efficiency. It is an essential ingredient of modern
GUI-based programs.

Points can be identified by coordinates. Lines and shapes can be
described by equations.

Path: A collection of lines and curves. Can be stroked(draw the path’s outline) and filled(color the interior region). 

Geometrical transformations – translation(moving), scaling, rotation, reflection and shearing(It slants the shape, shifting points horizontally or vertically, while keeping lines parallel. angles change but parallelism remains.) 
– can be applied easily to vector shapes.

SVG(Scalable Vector Graphics) provides a way of describing two-
dimensional vector graphics using XML.

Image size (bytes)=width (pixels) × height (pixels) × bytes per pixel

On the other hand, the same picture could be stored in the form of a
description of its component rectangles and their colors.
One possible format would be a short program in the
PostScript page description language, which just
occupies a total of 78 bytes:

## 3D

Although
objects can be drawn in 2D so that they appear as though they are 3D, if
you want to change the perspective or viewpoint in any way, you have to
redraw the object from scratch.

In 3D modeling, you can create 3D objects only once, then you can
view them from any angle or perspective without starting from scratch.

3D graphics programs calculate the proper highlight and shadow
information for a scene based on how you arrange the objects, colors,
textures, and lighting.

In actuality, 3D graphics should
be referred to as “two-dimensional representations of three-
dimensional objects”.

The spatial description and placement of **imaginary** 3D objects,
environments, and scenes with a computer system is called 3D
modeling.

3D space is a mathematically defined cube of cyberspace inside your
computer’s memory. Cyberspace differs from real physical space because
it is a mathematical universe that can exist only inside computer.

A line (a segment) means connecting-the-dots with two points.

A polyline is a continuous line that consists of multiple lines / segments. 

A polygon is a closed shape made by polylines.

A vertex is a point where any number of lines
come together and connect to each other.

Each one of the lines you draw forms a boundary, or edge of the polygon. 

And the area enclosed by the edges is called a **face**.

3D objects are made up of polygons, which are arranged by the computer into the form you desire.

The **viewpoint** represents the current vantage point of the user.
The viewing plane indicates the limits of the user’s view,
because only objects in front of that plane are visible.

world coordinate system, or global coordinate system. They remain the same regardless of
the viewpoint.

Screen coordinates, or view coordinates

Local coordinates

3D Display Modes:
- Bounding box
- Wireframe
- Hidden line
- Flat shaded
- Smooth shaded
- Smooth textured

## Working with 2D Shapes

Splines
- B-splines: control points usually do not lie on the curve, but influence its shape by the same weight. 
- Bezier: Defined by control points, but the endpoints are always on the curve.
- NURBS(Non Uniform Rational B-Splines): A more advanced version of B-splines. Control points can have different weights, letting you pull the curve more strongly in some areas.

2D Attach / Detach

2D Boolean operations

## Turning 2D Shapes into 3D Objects

Extrusions: simply pushing the 2D shape into the third dimension by
giving it a Z-axis depth.

Lathes: Rotating a 2D cross-section around an axis.
- closed 
- open
- partial: Lathes don’t have to be
a full 360° -- they could
just as easily be 90°, 180°, or 272°

Sweeps: Moving a 2D shape along a path.(i.e., single 2D cross-section that is extruded along a path)
- open
- helical
- closed

Skins: Wrapping surfaces across multiple cross-sections.

3D primitives

Advanced 3D primitives


### Main Modeling Techniques

Low-Poly modeling: Using fewer polygons for performance (important in games).

A poly count is the total number of polygons that make up a given
object.
- Most 3D engines prefer 4-sided polys – quads – but work fine
with 3-sided triangular polys as well. 
- The difference is that it takes two
triangular polys to make up a quad, so you can keep the poly count
lower by using quads rather than tris wherever possible.
- 3D engines have limits as to how many polygons they can move
around within a given time period, and this limit basically determines
how complex the geometry can be in a given scene.

Level of Detail (LOD): models are often created in two or three versions, each
with a different poly count, or level of detail 

**Mesh Tessellation** subdivides the faces in the selected area, dividing single
polygons into two or more to add more resolution to a surface.

**Mesh Optimization** can reduce the number of vertices and faces on an object
substantially without having too much of an impact on the rendered results.

Deform modifiers are transform settings (such as scale and twist) that are
applied to cross-sectional shapes as they are swept along a path.
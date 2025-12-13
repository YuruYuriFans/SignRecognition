## 1 Image Representation & Enhancement
### Application of Visual Perception
#### Edge detection
#### Anti-aliasing
#### Motion Blur
#### Image Compression
psycho-visually redundant. 

false contouring: a visual artifact in images and video where artificial, "ridge-like" outlines appear in smooth areas due to a lack of gray levels. 

### Digital Image Representation
Each element of the matrix array is called an image element,
picture element, pixel, or pel.

An image is referred to as a 2D light intensity function f(x, y) where

M rows, N columns, L discrete gray levels for each pixel $L = 2^k$

> (Due to processing, storage, and sampling hardware considerations, the number of gray levels typically is an integer power of 2.)

$b = M*N*k$

$2^k$ gray levels, referred as k-bit image.

#### spatial resolution

### Digital image processing

spatial domain methods: direct manipulation of pixels in an image

frequency domain methods: modifying the Fourier (or others) transform of an image.

#### Spatial domain processes
$g (x,y) = T [ f (x,y) ]$
- T is an operator on f, defined over some neighborhood of (x,y).

> Such approach is referred to as pixel group processing, mask processing or filtering.

pixel point processing

Linear and Nonlinear Operations
- Linear operations are exceptionally important in image processing because they
are based on a significant body of well-understood theoretical and practical results.
- Although nonlinear operations sometimes offer better performance, they are **not
always predictable**, and for the most part are not well understood theoretically.

### Image Enhancement
Image Enhancement: To process an image so that the result is suitable than the original image for
a specific application.
- The information of the image will not be increased but the chosen features.

#### Gray Level Transformations (GLT)
- Linear
  - negative: reverse the order of pixel intensities.
    - useful in displaying medical images (mammogram) and film processing
- Logarithmic
  - Dynamic range compression
- Power-law: $s = c r^\gamma$ where c and $\gamma$ are positive constants. 


Piecewise-Linear Transformation: complementary approach to the previous three basic transformations
- Advantage: It can be arbitrarily complex.
- Disadvantage: It requires considerably more user input.

##### contrast-stretching
idea behind contrast stretching is to **increase the dynamic range of the gray levels** in the image being processed.

in the extreme case, **thresholding function** that creates a binary image.

Window-Level Operation (a.k.a. Window-Center Adjustment)


##### Gray-level slicing
To highlight a specific range of gray levels in an image, such as
enhancing flaws in X-ray images.

Approach 1: Highlight range [A, B] of gray levels and reduce all others to a lower constant level

Approach 2: Highlight range [A, B] but preserve all other levels

#### Histogram Processing
Normalized Histogram

Four basic image types and their corresponding histograms

##### Histogram equalization

1. $s_k = T(r_k) = \sum_{j=0}^{k} \frac{n_j}{n}$
- rk is the k-th (k = 0, 1, …, L-1) grey level
- nk is the number of pixels in the image with that grey level
- n is the number of pixels in the image
- s has as many elements as the original histogram

2. scaling: $s_s = s * \text{maxGrayLevel}$, 4 bit is 15, because starting from 0

#### Spatial Filtering
Such approach is referred to as 
pixel group processing, mask processing or filtering.

The sub-image is called a (spatial) filter, mask, kernel, template, or window.

The values in a filter sub-image are referred to as **coefficients**, rather than pixels.

the term **spatial** filtering to differentiate this type of process
from the more **traditional frequency** domain filtering.

linear spatial filtering: **sum of products** of the filter
coefficients and the corresponding image
pixels in the area spanned by the filter
mask

> above process of linear filtering is similar to a frequency domain concept
called *convolution*, and is often referred to as “convolving a mask with an image”.
> > Similarly, filter masks are sometimes called *convolution masks*.

##### Image smoothing
https://www.geeksforgeeks.org/computer-vision/spatial-filtering-and-its-types/ 
Linear smoothing filters -- averaging filters(e.g. box filter, weighted filter)
- getting a gross representation of **objects of interest**
  - such that the intensity of smaller object blends with the background and larger objects
become “blob-like” and easy to detect.

Non-linear smoothing filters -- Order-statistics filters (e.g. median filter, max filter, min filter)

Median filters are particularly effective in the presence of impulse noise,
also called salt-and-pepper noise because of its appearance as white and
black dots superimposed on an image. (with considerably less blurring than linear
smoothing filters of similar size)

##### Image sharpening

Unsharp Masking 

1. Blur the original image
2. Subtract the blurred image from the original (the resulting difference is called the unsharp mask.)
3. Add the mask to the original to get the final sharpened signal

## 2 Morphological Image Processing
Mathematical morphology is very often used in applications where
shape of objects and speed are major issues.

Translation: $(A)_z$

Reflection: $\hat{B}$

dual view of binary images
- function 
- set

XOR: not both 1

NOT-AND: [NOT(A)] AND (B) in B not in A.

Primitive morphological operations
- dilation
- erosion

Dilation: 

Set B is commonly referred to as the structuring element

Structuring Elements (SEs):



Not inversed operations. Instead, e-d the result is a simplified and less detailed version of the original image.

Advanced morphological operations:

- Opening: Erosion followed by dilation. (Mnemonic:Opening Ed forum)
- Closing: Dilation followed by erosion

Alternative formula of opening:

Alternative (geometrically), closing

### Morphological Filters

Boundary Extraction: $\beta(A) = A - (A \ominus B)$

- first eroding A by B and then performing the set difference between A and its erosion.

## 3 Color Models & Color Image Processing

tristimulus theory

### Color model

color model (also called color space or color system)

RGB: computer color monitors, scanners.

CMYK: better for printing

HSV(also HSBrightness or HSI): humans describe and interpret color

#### RGB

Cube. Black at origin, white at the corner farthest from the origin.

color depth

color planes

> p13

### HSV

Hue: basic color, angle from 0 to 360.

Saturation: colors on the perimeter are
fully saturated, and the saturation decreases as you move to the center of
the wheel.

Value: can be thought of brightness, strictly speaking it is the maximum of RGB values. Third axis, extends to a cylinder.

Conceptual relationships between the RGB and HSV color models:

- Take the color cube and stand it on the black (0,0,0) vertex, with the white
vertex (1,1,1) directly above it.

> p21

HSV not commonly used to store, but more commonly for manipulating colors.

### Pseudo-color Image Processing

Pseudocolor (false color) image processing consists of assigning
colors to gray values based on a specified criterion.
- Principal use

Intensity Slicing Coding

Sample colorize

### Full-Color Image Processing

Category 1: we process each component image(RGB/HSV) individually and then
form a composite processed color image from the individually processed
components.
- HSV manipulation, Histogram equalization


Category 2: we work with color pixels directly. 
- Spatial filtering

Color manipulation
- Gamma correction, $O = I^{1 / \gamma}$
  - no matter what gamma
correction you apply to an image, pixels with a value of 0 or 1.0 will
remain unchanged. The only effect that the gamma operator has will
be on non-black and non-white pixels.
    - It makes the image tend to look more natural.
- Contrast: multiply and substract
> the generally accepted convention is that
applying a gamma of 0 to an image will produce a black frame.

### Geometric transformation
Image enhancement is largely a **subjective** process, while image
restoration and retouching are for the most part an **objective** process,
such as geometric transformation.


Working resolution:

Panning: offset X and Y

Rotation

Scaling

Warping: image were printed on a thin sheet of flexible rubber.
This rubber sheet can be pushed and pulled by various amounts in various areas until the
desired result is obtained.

### Image Compositing

aka matting in the film industry.

main purposes

Mattes.

fourth channel, matte channel or alpha channel.

12 Compositing Operations

OVER

MIX
- MV refers to the mix value, (the percentage of the first image A that we will be using)

mask

## 4 Video Data Representation & Processing

## 5 Audio Data Representation & Coding

Audiosonic: 20Hz to 20kHz, human ear detection. the upper limit decreases fairly rapidly with increasing age.

Periodic:

Non-periodic:

Sampling rates(aka resolutions):
- 48kHz
- 44.1kHz
- 22.05kHz
- 11.025kHz

Quantization:
- 16 bits, 65536 quatization levels
- 8-bit: minimum acceptable

Dithering: injection of noise.

If a compromise must be made, the effect on quality of
reducing the sampling size is more drastic than that of
reducing the sampling rate 

Audio file size: if rate $r$ Hz and sampling size $s$ bits, each second occupy $r*s/8$ bytes.
- based on a single channel, but audio is almost always recorded in stereo, so the estimates should double.

### Sound Compression

#### PCM coding

##### Nonlinear PCM coding

To get a wider dynamic range, you can use nonlinear encodings.

Human hearing is relatively insensitive to small errors in loud sounds but more
sensitive to similar errors in quiet sounds.

The nonlinear PCM coding works well for compression. It consists of using the
available bits more efficiently; you use **more bits for quiet sounds** in which the
data loss is more audible.

$\mu$-law and $A$-Law, two most common non-linear PCM coding. 

They all use logarithmic formulas to convert 16- or 12-bit linear PCM
samples into 8-bit codes.

Also called Logarithmic Compression:

- Most sound consists predominantly of small samples. These encodings
provide more accuracy for these common values at the cost of less
accuracy for relatively infrequent large samples;

- Human hearing is logarithmic; a change in the intensity of a quiet sound
is more noticeable than a similar change in the intensity of a loud sound.

Essentially, logarithmic encoding provides more accuracy where that
accuracy is most audible.

Linear PCM will always be preferred for manipulating sound.
Whenever you mix two sounds by adding or adjust volume by
multiplication, you are assuming your sound format is linear.

Nonlinear PCM coding is more suitable for transferring and
storing sound.

#### DPCM coding

If your sampling rate is sufficiently high, the differences between successive
samples are likely to be small.

As a result, you may be able to store the
same audio signal using fewer bits per sample by storing the **differences between successive samples** 
rather than the samples themselves.

> sometimes called delta modulation. It is a simple way to achieve modest compression.

In DPCM coding, in order to keep the differences small, you need to use
a higher sampling rate(such that the differences are more likely to be small), but that negates the benefits of smaller samples.


DPCM codng:
Fbonacci:
Exponential:

Note that both of these difference-encodings provide many small values
(small differences are generally more common) and a few larger values.


#### AdaptiveDPCM coding

choose a set of possible differences based on prior data.

This often takes the form of a variable scaling factor. quantized_difference = difference / scale_factor.

- If the scaling factor is
small, you can represent small differences but not large ones;

- if the scaling
factor is large, you can represent large differences but not small ones.

By adjusting the scaling factor for **different sections** of the sound, you can
provide better quality than plain DPCM.

ADPCM techniques are a popular means of audio compression. They are
easy to program, fast, and can provide around 4:1 compression with
reasonable sound quality.

#### Predictor-Based Compression
data modeling: gussing the next element by the previous ones.

Because of random errors and other factors, it is difficult to guess exactly
right.


#### Perceptually-Based Compression

Two phenomena in particular cause some sounds not to be heard,
despite being physically present.
1. too quite to be heard
2. obscured by other sounds

The perception sensitivity we call loudness is **not linear** across all frequencies.
- most sensitive: 700 ~ 6.6kHz
- The human hearing system responds much better to the mean frequency
range than it does to low and very high frequencies.


The (absolute) threshold of hearing is the
minimum level at which a sound can be
heard. It varies non-linearly with frequency.

psycho-acoustical model 

Any sound that lies within the masking curve will be inaudible, even
though it rises above the un-modified threshold of hearing.

#### Subband Coding

One way to improve differential techniques is to divide the sound into
two or more frequency ranges, or sub-bands, and compress each
one separately.

Subbands near the center of the
human hearing range can then be preserved while those to which human
hearing is less sensitive can be treated less carefully, even dropped entirely.

Subband coding can typically compress PCM audio data by a factor of
10 to 20.

#### Progressive Compression

Sometimes your bandwidth of a network connection can vary widely.

### Speech

Speech technology is becoming increasingly important in both personal and
enterprise computing as it is used to improve existing user interfaces and
to support new means of human interaction with computers.

Speech output

Speech input

Speech Synthesis(text-to-speech) conversion

Speech recognition

#### Speech Synthesis

1. Structure analysis: where paragraphs, sentences and other structures start and end using punctuation and formatting data
2. Text pre-processing: Analyze the input text for special constructs of the language. (abbreviations, acronyms, dates, times, numbers, currency amounts, email addresses)
3. Text to phoneme conversion: 
4. Prosody analysis
5. Waveform production

## 6 2D & 3D Computer Graphics

## 7 computer animation & multimedia authoring

## 8 image data analysis & retrieval

Text based: conceptual level

Content based: perceptual level, with objective and
quantitative measurements of the visual content and integration of image
processing, pattern recognition, and computer vision.

A global descriptor uses the visual features of the whole image.

A local descriptor uses the visual features of regions or objects to
describe the image content, with the aid of region segmentation and
object segmentation techniques.

### Similarity measures

Minkowski distance: feature vectors independent of each other, of equal importance.
- p = 2, Euclidean distance
- p = 1, Histogram intersection

Quadratic-Form (QF) Distance: $D(I_Q, I_D) = \sqrt{(F_I_Q - F_I_D)^T A(F_I_Q - F_I_D)}$

Some colors are similar, but treated differently by Minkowski distance. 

$A$: similarity matrix
- $a_{ij}$ near 1 = very similar, near 0 = very different color bin i to j.

Why QF is good
- It captures cross-bin similarity between colors
- It produces more perceptually meaningful results
- Used commonly in color histogram–based image retrieval

### Indexing scheme

Because the feature vectors of images tend to have high dimensionality
and therefore are not well suited to traditional indexing structures,
**dimension reduction** is usually used before setting up an efficient
indexing scheme, by using PCA (principal component analysis), KL
(Karhunen-Loeve) transform, or neural network techniques.

### Color

Color stimuli: 

Most commonly used color descriptors: color moments, color histogram,
color coherence vector (CCV) and color correlogram.

The **hue** is invariant to the changes in illumination and camera
direction and hence more suited to object retrieval.

#### Color moments 
successful especially when the image contains only objects

Since only 9 (three moments for each of the three color components)
numbers are used to represent the color content of each image, color
moments are very compact representations compared to other color
features.
- also lower discrimination power, first pass to narrow down the search space.

#### color histogram
more bins = more discrimination power, but also more computational cost and inappropriate for building indexes.
- Furthermore, a very fine bin quantization does not necessarily improve
the retrieval performance in many applications.

Reducing the number of bins:
- Down-sampling the color depth / Quantization of the color space;
- Use the bins that have the largest pixel numbers (may even enhance since small bins are likely to be noisy).
- Clustering methods – determine the K best colors and then calculate the
number of pixels that fall in each of the K best colors.


#### color coherence vector

#### color correlogram

### Texture

Texture is a powerful discriminating visual feature which has been widely
used in pattern recognition and computer vision for identifying visual
patterns with the local intensity variations, based on different texture
properties such as smoothness, coarseness, regularity, and homogeneity.

Structural methods, including Morphological Operator and Adjacency
Graph methods, describe texture by identifying structural primitives and
their placement rules. They tend to be most effective when applied to
textures that are very regular.

Statistical methods, including Co-occurrence Matrices, characterize texture by the statistical distribution of the image intensity.

#### Co-occurrence Matrices

this configuration varies rapidly with distance in fine textures and slowly in coarse textures.

Step-1: Construction of co-occurrence matrices

For computational efficiency, the number of gray levels can be reduced by binning, i.e., a simple
procedure in which the total range of values is divided by a smaller amount – the required number
of bins, thus “shrinking” the co-occurrence matrix.

On their own, these co-occurrence matrices do not provide any measure of texture that can easily
be used as texture descriptors. The information in the matrices needs to be **further extracted** as a
set of feature values.

Step-2: Calculation of statistic quantities from the co-occurrence matrices

Totally 14 second-order statistic quantities called **Haralick texture features** can be computed out
of the coefficients.
- to compare images of different sizes, matrices are normalized by dividing each coefficient in a matrix by the sum of all elements.

F1(energy): a measure of textural uniformity in the image. Higher energy → pixels change slowly → texture is smooth, uniform.

F2(contrast): measuring the amount of local variations in the image. High contrast → big differences between neighbouring pixel intensities

F5(inverse): measuring image homogeneity. High IDM → texture varies slowly and smoothly. Similar to energy but more sensitive to local smoothness.

F9(entropy):measuring the disorder of the image

#### Tamura features

Designed in
accordance with psychological studies on the human perception of texture.
- Coarseness: Measures the granularity of a texture.
- Contrast: Measures the intensity or brightness variations in an image.
- Directionality: Captures the dominant orientation of textures.
- Line-likeness: Describes the prevalence of "line-like" patterns.
- Regularity: Measures the periodicity or repetitive nature of a texture.
- Roughness: Often defined as the combination of coarseness and contrast.

### Shape

The shape of an object or region refers to its profile and physical
structure.

Compared with color and texture features, shape features are
usually described **after images segmented** into regions or
objects.

Since robust and accurate image segmentation is difficult to achieve,
the use of shape features for image retrieval has been **limited to
special application** where objects or regions are readily available.

A good shape representation feature for an object should be invariant to
translation, rotation, scaling, affine transformation (such as stretching
and squeezing).

Bounding Box: feature descriptors
1. Area of Bounding Box
2. Ratio of edges (HL/VL)

Chain codes:  generated by
following a boundary in a clockwise direction and
assigning a direction to the segments connecting
every pair of pixels.

Limitations: sensitive to noise, Raw chain code is usually too long, Resampling or smoothing is needed

Fitting Line Segments:

#### Geometry features

Perimeter measurement

Area attribute

Roundness, or compactness $\gamma = \frac{\text{(perimeter)}^2}{4\pi(\text{area})}$. For a disc, the roundness parameter is minimum and equals 1.

#### Medial Axis Transformation (MAT)

An important approach to representing the structural shape of a plane region is to
reduce it to a graph.

Prairie Fire Concept. The MAT of the region is the set of points reached by more than
one fire front at the same time.

#### Syntactic Representation
simply a string of symbols, each representing a primitive. The syntax allows a **unique** representation and interpretation of the string.

The design of a syntax that transforms the symbolic and the syntactic
representations back and forth is a difficult task. It requires specification of a
**complete** and **unambiguous** set of rules, which have to be derived from the
understanding of the scene under study.

### Spatial relationships

Directional relationships

Topological relationships(neighborhood and incidence):

Attributed Relational Graphs (ARGs)
- objects: nodes
- relationships: arc between nodes

p = perimeter

rd = relative distance: minimum distance between their surrounding contours

rp = relative position: inside or outside

Region Relationship Matrix

$\pi_{ij}$ is the common perimeter between i and j, $\pi_i$ and $\pi_j$ are individual perimeters respectively.

$d_{ij} = ||v_i - v_j||$ is the distance between the centroids of i and j.

$\theta_{ij}$ the angle between the horizontal (column) axis and the line joining the
centroids

class functions divided into 3 relationship groups
1. perimeter class: DISjoined, BORdering, INVaded_by, SURrounded_by
2. distance class: NEAR, FAR
3. Orientation class: RIGHT, LEFT, ABOVE, BELOW



## 10 video data analysis & retrieval

Case 1

Case 2

Case 3

Content-based video retrieval (CBVR) systems appear like a
natural extension of content-based image retrieval (CBIR)
systems.

temporal information

structural organization of the video sequence

complexity of the querying systems. Query-by-example (QBE) far more complicated to adapt into the context of video sequences.

Accessing Video Content

### Video Content Analysis

#### Shot boundary detection

Automatic shot boundary detection techniques can be classified
into five categories: pixel based, statistics based, transform based,
feature based, and histogram based.

The histogram-based approach is the most popular. It uses
histograms of the pixel intensities as the measure, and achieves a
good tradeoff between accuracy and speed.


#### Key frame extraction

After the shot boundaries are detected,
corresponding key frames can then be extracted.
- Simple approaches may just extract the first and last frames of each shot as
the key frames.
- More sophisticated key frame extraction techniques
are based on visual content complexity indicators, shot activity
indicators, and shot motion indicators.

Temporal variants

Clustering

Curve splitting

### Video Content Representation / Feature

1
- Sequential key frame representation / feature
- Shot-based representation / feature
- Scene-based representation / feature
- Temporal slice based representation / feature
- Object-based representation / feature

When the video clip is long, Sequential key frame representation does not
scale, since it does not capture the embedded information
within the video clip, except for time.

Shot-based representation includes statistical motion
measures, and the temporal mean and variance of color
features over a shot, to provide information about activities,
and motion complexity and distribution that might be useful
in queries.

Video ToC (table of
content) construction at the scene level is thus of fundamental
importance to video browsing and retrieval.


In the object-based representation, the main attributes attached to key or
dominant objects are motion, shape and life cycle, as well as other
image features such as color and texture.


## 11 Image Coding & Compression

The aim of digital image compression is to represent an image
with the smallest possible number of bits, for reducing the
storage required to save an image, or reducing the bandwidth
required to transmit it.

Compression ratio: $C_R = n_1 / n_2$, $n_1, n_2$ denote the number of information-carrying units in two
data sets that present the same information.
- $C_R > 1$ for compression to work (sanity check)

relative data redundancy: $R_D = 1 - 1 / C_R$

### three basic data redundancies

#### Coding Redundancy

code word: Each piece of information or event is assigned a sequence of code symbols
- length: number of symbols in each code word is its


coding redundancy

$r_k$: grey levels of an image

$p_r(r_k) = n_k / n$: probability of each grey level, where $k = 0, ..., L - 1$

$l(r_k)$: number of bits used to represent each $r_k$

Entropy of the source:

total number of bits required to code an $M \times N$ image: $M*N*L_{avg}$

variable-length coding


#### Interpixel redundancy

inter-pixel redundancy, or spatial redundancy, geometric redundancy: each pixel's value can be predicted from its neighbors, so there is redundant information between pixels.

mappings: transformation using the difference between adjacent pixels
- reversible mappings

$g_i$: grey level

$w_i$: run length (how many times it repeats consecutively)

run length encoding(RLE) is reversible mapping.

#### Psycho-visual redundancy

psycho-visually redundant: Certain information simply has less relative importance than other
information in normal visual processing
- fundamentally different because the information itself is not essential for normal
visual processing

quantization: elimination of psycho-visually redundancy
- irreversible operation, results in lossy data compression.

IGS (Improved Gray-Scale) quantization: reduces visible false-contouring by adding small pseudorandom variations from neighboring pixels' low-order bits **before** quantizing.

> Low-order bits are the least significant bits (the ones on the right side) of a binary number.

### Image Compression Models

error free or information preserving, ig not, some level of distortion in the reconstructed image.



- Encoder
  - Source encoder: removes input redundancies (coding, inter-pixel, psycho-visual)
  - Channel encoder: noise immunity
- Channel
- Decoder
  - Channel decoder
  - Source decoder

#### Source Encoder
three independent operations and 
each operation is designed to reduce one of the three redundancies.

Mapper: inter-pixel 
- generally reversible, may or may not reduce direct amount

quantizer: psycho-visual
- irreversible, must be omitted for error-free compression

symbol encoder: coding
- reversible

> All three operations are not necessarily included in every compression system.

#### Source decoder

Symbol decoder

Inverse mapper

#### Error-Free (Lossless) Compression

modest compression ratios of the order of 2 to 10

##### Variable-Length Coding
Huffman coding yields the minimum number of code symbols per source symbol. 5 steps.

block code: simple look-up table.
- instantaneous: decodable without referencing succeeding symbols.
- uniquely decodable: any string can be decoded in only one way.

In general, Huffman coding will give low
compression ratios for flat histogram
images and higher compression ratios for
sharp histograms.

##### Lossless Differential Coding
reduce interpixel redundancy by storing differences between neighboring pixels instead of the raw pixel values.

differences tend to be small and cluster around zero, creating a sharper histogram (lower entropy) than the original image for better Huffman coding performance.

##### LZW Coding
LZW (Lempel-Ziv-Welch) coding losslessly targets inter-pixel redundancies, used in GIF, TIFF, PDF.

without knowing probabilities, builds the dict dynamically.

#### Near-lossless compression

background removal

#### Lossy Compression
Significantly:
- recognizable monochrome images with 100:1, virtually indistinguishable from the originals at 10:1 to 50:1
  - where lossless seldom exceeds 3:1

##### Quantization

##### Predictive Coding

eliminating the inter-
pixel redundancies of closely spaced pixels by extracting and coding
only the new information in each pixel.

| Feature               | Lossless Differential Coding              | Predictive Coding                                                             |
| --------------------- | ----------------------------------------- | ----------------------------------------------------------------------------- |
| **Type**              | Always lossless                           | Can be lossless or lossy                                                      |
| **Prediction method** | Simple difference (usually left neighbor) | Uses one or more neighbors (e.g. A, B, C)                                     |
| **Output data**       | Pixel-to-pixel differences                | Prediction errors                                                             |
| **Goal**              | Reduce redundancy                         | Reduce redundancy, possibly compress more                                     |
| **Example**           | ( d(x, y) = I(x, y) - I(x-1, y) )         | ( e(x, y) = I(x, y) - \hat{I}(x, y) ), where ( \hat{I} ) is a predicted value |

## 12 Video Coding & Compression
<!-- 31 to 42, 49 to 77 -->


Redundancy exists in a video sequence in two forms: spatial(intra-frame) and temporal(inter-frame).

- spatial: within a single frame
- temporal: beteen consecutive frames

### Spatial / Intra-frame Compression

Since video is just a sequence of images, image compression
techniques are directly applicable to video frames.

### Temporal / Inter-frame Compression

frame differencing, an extension of the basic differential pulse code modulation (DPCM) coding techniques: to code the difference between one frame and
the next. 

At the initial point, a frame must be coded without frame differencing and
using only spatial / intra-frame coding techniques. Such a frame is
commonly referred to as an intra-coded frame (I frame).

Because I frames do not take advantage of inter-frame redundancy,
they consume more bits than predictive frames (P frames) of
comparable quality.

Motion Compensation (M.C.)
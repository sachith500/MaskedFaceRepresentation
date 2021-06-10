# Contrastive Face Representation


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Masked face recognition focuses on identifying people using their facial features while they are wearing masks. We introduce benchmarks on face verification based on masked face images. 


## Data and Ethics
All models, code and research presented in this repo is provided for reproducibility purposes. Our work is intended for use for helping security-related applications while maintaining COVID-safe protocols (keep your mask on at airport security to maintain a lower chance of spreading COVID) and not for invading individual privacy.
We invite researchers to consider the broader implications of perfecting state of the art result work in areas such as masked identification. We believe everyone has the right to their privacy and researchers should consider the broader implications of their work and potential for misuse.

The accuracy and performance of models released in this repo should be sufficient for health-related applications (such as safety protocols) but would not (we hope) be usable for tracking an individuals' movements.

If you are interested in using this work for COVID-Safety applications we are happy to consult on a pro bono basis.

## Features
- Generate Synthetic masks to CelebA, Fei Face, georgia_tech, SoF, YoutubeFaces and LFW datasets 
- Apply synthetic mask to a face image
- Apply synthetic masks to a folder of images
- [NeurIPS2021] Regenerate results in the paper : Multi-Dataset Benchmarks for Masked Identification using Contrastive Representation Learning
    - Regenerate benchmark 1 results from the models
    - Regenerate benchmark 2 results from scores files
    - Regenerate benchmark 2 results from the models

## Download the Benchmark Datasets
| Dataset Name | Website |Instructions|Download URL|
| ------ | ------ |------|------|
| celeba | [Large-scale CelebFaces Attributes (CelebA) Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |Follow the downloading instructions|https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg|
| fei_face_frontal | [FEI Face Database](https://fei.edu.br/~cet/facedatabase.html) |Download both folders and merge into fei_face_frontal folder |https://fei.edu.br/~cet/frontalimages_manuallyaligned_part1.zip https://fei.edu.br/~cet/frontalimages_manuallyaligned_part2.zip |
| fei_face_original |[FEI Face Database](https://fei.edu.br/~cet/facedatabase.html)  |Download and merge all the folders into fei_face_original folder|https://fei.edu.br/~cet/originalimages_part1.zip https://fei.edu.br/~cet/originalimages_part2.zip https://fei.edu.br/~cet/originalimages_part3.zip https://fei.edu.br/~cet/originalimages_part4.zip|
| georgia_tech | [Georgia Tech face database](http://www.anefian.com/research/face_reco.htm) |Download the zip and unzip into georgia_tech|http://www.anefian.com/research/gt_db.zip|
| sof_original | [Specs on Faces (SoF) Dataset](https://sites.google.com/view/sof-dataset) |Download original images|https://drive.google.com/file/d/1ufydwhMYtOhxgQuHs9SjERnkX0fXxorO/|
| youtube_faces | [YouTube Faces DB](https://www.cs.tau.ac.il/~wolf/ytfaces/) |Follow the downloading instructions|https://www.cs.tau.ac.il/~wolf/ytfaces/ |
| lfw | [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) |Download all images aligned with deep funnelling|http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz|

### Downloading instructions
##### celeba
1. Go to the google drive link
2. Go inside img_align_celeba_png folder
3. Download img_align_celeba_png.7z001 - img_align_celeba_png.7z016 all files
4. Combine them and unzip it using 7zip
5. Unzip the file to celeba/images folder
6. Download annotations identity_CelebA.txt from the following link https://drive.google.com/drive/u/1/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ
7. Place annotations identity_CelebA.txt under celeba/

##### youtube_faces
1. Fill the form by providing your name and email.
2. Get the username and password to login to the provided website.
3. Click on frame_images_DB then go to YouTube Faces(YTF) data set.
4. Download YoutubeFaces.tar.gz
5. Unzip under youtube_faces

### Folder Structure after downloading the datasets
base_folder : Base folder for all the databases
- celeba
    - images
    - identity_CelebA.txt
- fei_face_frontal
- fei_face_original
- georgia_tech
- lfw
- sof_original
- youtube_faces
## Generate Synthetic masks to benchmark datasets


Install python 3.6. Then run the following command to install the requirements.

```sh
pip install -r requirements.txt
```

To apply synthetic masks on the datasets

```sh
python generate_mask_to_datasets.py --base_folder ../base_folder --new_dataset_folder ../benchmark_dataset
```

## Summary of the datasets
|Dataset |Unmasked Identities/ ImagesMasked |Identities/ Images|
| ------ | ------ | ------ |
|CelebA |10177/202,599|10174/197,499|
|FEI Face |200/1,177|200/1,177|
|Georgia Tech |50/750|50/750|
|SoF |93/1,443|90/1,393|
|YouTube Faces |1595/20,252|1589/19,960|
|LFW |5749/13,167|5718/13,138|

## NeurlIPS2021 Experiments

Our network architecture (Siamese Network Architecture)
![](NeurIPS2021/graphs/masknn.jpg?raw=true)

### Benchmark 1

#### Results
| |ImageNet| |VGGFace2 | |Proposed|
| ------ | ------ | ------ | ------ |------ |------ |
|Dataset|VGG19|MobileNet|SENET|VGG16|ResNet50|
|fei_face_original|0.363|0.356|0.49|0.304|0.031|
|georgia_tech|0.323|0.416|0.483|0.431|0.097|
|sof_original|0.476|0.389|0.415|0.365|0.169|
|fei_face_frontal|0.357|0.171|0.424|0.143|0|
|youtube_faces|0.424|0.394|0.468|0.385|0.115|
|lfw|0.361|0.449|0.469|0.372|0.142|
|in_house_dataset|0.288|0.244|0.425|0.288|0.038|

#### Regenerate benchmark 1 results

```sh
cd NeurIPS2021
python regenerate_benchmark_1_from_models.py --base_folder ../base_folder
```
### Benchmark 2
#### Results
|Dataset|Exp1|CP1|CP2|FT1|FT2|FT3|Ensemble|
| ------ | ------ | ------ | ------ |------ |------ |------ |------ |
|fei_face_original|0.073|0.016|0.015|0.01|0.016|0.011|0.009|
|georgia_tech|0.207|0.041|0.055|0.06|0.059|0.058|0.048|
|sof_original|0.187|0.073|0.071|0.058|0.069|0.067|0.061|
|fei_face_frontal|0|0|0|0|0|0|0|
|youtube_faces|0.156|0.053|0.051|0.042|0.056|0.046|0.041|
|lfw|0.219|0.101|0.09|0.091|0.11|0.093|0.084|
|in_house_dataset|0.038|0.031|0.013|0.013|0.013|0.019|0.013|


#### Regenerate benchmark 2 results

```sh
cd NeurIPS2021
python regenerate_experiment_2_results_from_models.py --base_folder ../base_folder
```
### Overall benchmark:  results of Exp1(trained on CelebA for 1015k steps) and Ensemble(trained on 4 datasets) on the synthetic unmasked-masked datasets generated

| |Exp1 (Celeb Only)| |Ensemble (4 datasets)| |
| ------ |------ |------ |------ |------ |
|Dataset|EER|FRR100|EER|FRR100|
|fei_face_original|0.089984|0.638723|0.008984|0.015968|
|georgia_tech|0.142884|0.93014|0.047976|0.245509|
|sof_original|0.195122|0.762745|0.061094|0.178431|
|fei_face_frontal|0.071429|0.2|0|0|
|youtube_faces|0.142902|0.904|0.040948|0.208|
|lfw|0.17788|0.976048|0.08387|0.229541|
|in_house_dataset|0.075|0.15|0.0125|0.025|

#### Regenerate overall benchmark

```sh
cd NeurIPS2021
python regenerate_table_6_results_from_models.py --base_folder ../base_folder
```

### FPR/TPR curve, for all datasets using CP1 model
The relative difficulty of different datasets can be visualized based on CP1 model.
![](NeurIPS2021/graphs/FPR_TPR_all.png?raw=true)

### Results

### Additional Results


#### FNMR1000 rates for experiment 2


| |CP1|CP2|FT1|FT2|FT3|ENSEMBLE|
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | 
|fei_face_original|0.021956|0.0499|0.045908|0.073852|0.02994|0.043912|
|georgia_tech|0.311377|0.277445|0.421158|0.377246|0.369261|0.245509|
|sof_original|0.533333|0.462745|0.580392|0.52549|0.57451|0.407843|
|fei_face_frontal|0|0|0|0|0|0|
|youtube_faces|0.41|0.446|0.292|0.374|0.286|0.318|
|lfw|0.325349|0.437126|0.43513|0.303393|0.373253|0.46507|
|in_house_dataset|0.05|0.05|0.025|0.1|0.05|0.025|



#### FNMR0 rates for experiment 2

| |CP1|CP2|FT1|FT2|FT3|ENSEMBLE|
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | 
|fei_face_original|0.021956|0.0499|0.045908|0.073852|0.02994|0.043912|
|georgia_tech|0.311377|0.277445|0.421158|0.377246|0.369261|0.245509|
|sof_original|0.533333|0.462745|0.580392|0.52549|0.57451|0.407843|
|fei_face_frontal|0|0|0|0|0|0|
|youtube_faces|0.41|0.446|0.292|0.374|0.286|0.318|
|lfw|0.325349|0.437126|0.43513|0.303393|0.373253|0.46507|
|in_house_dataset|0.05|0.05|0.025|0.1|0.05|0.025|



These results indicate the performance of the explored models on the different datasets

#### Evaluation on using more data points for analysis from each dataset.
We run the multi dataset-trained models through a larger test-set from each dataset. We draw a single image at random for each identity to serve as the reference, and a single authentic and imposter image to act as probes for each identity. So, this test uses 2 tests per identity for each identity in each dataset. We present the 1000-sample test set in the paper as a) the results are easily reproducible and b) the size of the test is fixed (except for in-house data, which is only used for validation purposes).

#### Results on all identities (2*n pairs of images with n identities)

#### Result on 1000 pairs

| |EX1|CP1|CP2|FT1|FT2|FT3|Ensemble|
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
|fei_face_original|0.089984|0.015988|0.01498|0.009984|0.015992|0.010988|0.008984|
|georgia_tech|0.142884|0.04096|0.054958|0.059912|0.058926|0.057922|0.047976|
|sof_original|0.195122|0.073194|0.070898|0.058456|0.069386|0.066527|0.061094|
|fei_face_frontal|0.071429|0|0|0|0|0|0|
|youtube_faces|0.142902|0.052902|0.050948|0.041936|0.05597|0.04593|0.040948|
|lfw|0.17788|0.100892|0.089864|0.09087|0.109926|0.092878|0.08387|
|in_house_dataset|0.075|0.03125|0.0125|0.0125|0.0125|0.01875|0.0125|



## License

MIT

## Citations

If you use the facial recognition work, please cite:

TODO: add link to final publication on recognition.

If you use the race, sex or age prediction work, please cite:

TODO: add link to final publication.

## Acknowledgements

This project is supported by National Health and Medical Research Grant GA80134. 
This research was undertaken using the LIEF HPC-GPGPU Facility hosted at the University of Melbourne.
This Facility was established with the assistance of LIEF Grant LE170100200. 
This research was undertaken using University of Melbourne Research Computing facilities established by the Petascale Campus Initiative.




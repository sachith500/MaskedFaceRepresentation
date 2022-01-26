# Masked Face Representation


[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Masked face recognition focuses on identifying people using their facial features while they are wearing masks. We introduce benchmarks on face verification and facial attribution prediction based on masked face images corresponding to multiple published articles.


## Data and Ethics
All models, code and research presented in this repo is provided for reproducibility purposes. Our work is intended for use for helping security-related applications while maintaining COVID-safe protocols (keep your mask on at airport security to maintain a lower chance of spreading COVID) and not for invading individual privacy.
We invite researchers to consider the broader implications of perfecting state of the art result work in areas such as masked identification. We believe everyone has the right to their privacy and researchers should consider the broader implications of their work and potential for misuse.

The accuracy and performance of models released in this repo should be sufficient for health-related applications (such as safety protocols) but would not (we hope) be usable for tracking an individuals' movements.

If you are interested in using this work for COVID-Safety applications we are happy to consult on a pro bono basis.

## Features
- Masked face identification
    - Generate Synthetic masks to CelebA, Fei Face, georgia_tech, SoF, YoutubeFaces and LFW datasets 
    - Apply synthetic mask to a face image
    - Apply synthetic masks to a folder of images
    - [DICTA2021] Regenerate results in the paper : Multi-Dataset Benchmarks for Masked Identification using Contrastive Representation Learning
        - Regenerate benchmark 1 results from the models
        - Regenerate benchmark 2 results from scores files
        - Regenerate benchmark 2 results from the models
- Masked face privacy
    - Generate Synthetic masks to UTKFace dataset
    - Split the dataset with proper distribution among training, validation and test datasets
    - [AJCAI2021] Regenerate results in the paper : Does a Face Mask Protect my Privacy?: DeepLearning to Predict Protected Attributes fromMasked Face Images
        - Regenerate the overall accuracy for models
        - Regenerate the results of masked and unmasked faces


## Masked face identification

Conference Venue: DICTA2021

### Download the benchmark datasets
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
base_folder : Base folder for all the datasets
- celeba
    - images
    - identity_CelebA.txt
- fei_face_frontal
- fei_face_original
- georgia_tech
- lfw
- sof_original
- youtube_faces


### Generate Synthetic masks to benchmark datasets


Install python 3.6. Then run the following command to install the requirements.

```sh
pip install -r requirements.txt
```

To apply synthetic masks on the datasets

```sh
python generate_mask_to_datasets.py --base_folder ../base_folder --new_dataset_folder ../benchmark_dataset
```

### Summary of the datasets for DICTA2021
|Dataset |Unmasked Identities/ Images |Masked Identities/ Images|
| ------ | ------ | ------ |
|CelebA |10177/202,599|10174/197,499|
|FEI Face |200/1,177|200/1,177|
|Georgia Tech |50/750|50/750|
|SoF |93/1,443|90/1,393|
|YouTube Faces |1595/20,252|1589/19,960|
|LFW |5749/13,167|5718/13,138|

### Experiments

Our network architecture (Siamese Network Architecture)
![](DICTA2021/graphs/masknn.jpg?raw=true)

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
cd DICTA2021
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
cd DICTA2021
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
cd DICTA2021
python regenerate_table_6_results_from_models.py --base_folder ../base_folder
```

### FPR/TPR curve, for all datasets using CP1 model
The relative difficulty of different datasets can be visualized based on CP1 model.
![](DICTA2021/graphs/FPR_TPR_all.png?raw=true)

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

## Masked face privacy

Conference Venue: AJCAI2021

### Download the benchmark datasets

| Dataset Name | Website |Instructions|Download URL|
| ------ | ------ |------|------|
| UTKFace | [UTKFace Dataset](https://susanqq.github.io/UTKFace/) |Follow the downloading instructions|https://drive.google.com/file/d/0BxYys69jI14kb2o4ajJwQ3FOUm8/view?usp=sharing&resourcekey=0-wviJQhRUIJUlUjFc86H0kg https://drive.google.com/file/d/0BxYys69jI14kNEt1SnNJN1Z2WWc/view?usp=sharing&resourcekey=0-iUnGnz7QyDHeZOYdHcMm4A https://drive.google.com/file/d/0BxYys69jI14kVkVTZHZHa21zUXM/view?usp=sharing&resourcekey=0-AzmPdtIpMfLjjRox3dEs-g|

### Downloading instructions
##### UTKFace
1. Download the dataset using the given URLs.
1. Download part1.tar.gz, part2.tar.gz and part3.tar.gz
1. Unzip the files and get the images to UTKFace folder

### Folder Structure after downloading the datasets
base_folder : Base folder for the dataset (This is where you unzip and copy UTKFaces folder)
- UTKFace

### Generate Synthetic masks to benchmark datasets

Install python 3.6. Then run the following command to install the requirements.

```sh
pip install -r requirements.txt
```

To apply synthetic masks on the datasets

```sh
python generate_mask_to_datasets.py --base_folder ../base_folder --new_dataset_folder ../benchmark_dataset
```

### Summary of the datasets for AJCAI2021
|Dataset |Unmasked Images |Masked Identities/ Images|
| ------ | ------ | ------ |
|UTKFace |24,107|23,004|

#### Age buckets used for classification

|Age group |Age range |
| ------ | ------ | 
|Baby |0-3 years|
|Child |4-12 years|
|Teenagers |13-19 years|
|Young |20-30 years|
|Adult |31-45 years|
|MiddleAged |46-60 years|
|Senior |61 and above years|

#### Overall prediction comparison with confusion matrices

![](AJCAI2021/data/cf-all.png?raw=true)

### Overall accuracy for models

#### Results
| Method|Sex accuracy| Race accuracy |Age Accuracy ||
| ------ | ------ | ------ | ------ |------ |
| | | |MAE|RMSE|
| Previous implementations | 0.9374 | - | - |- |
| Our method with transforms | **0.9401** | **0.8220** | 6.2788 |8.4836 |
| Our method without complex transforms | 0.9361 | 0.8134 | **6.2168** |**8.3372**|

#### Regenerate attribute prediction accuracy results with transforms

### Results for masked and no mask faces

#### Results
| |Unmasked face - SOTA| Masked Face (Random Split) |Masked Face (Uniform Split) |
| ------ | ------ | ------ | ------ |
|Sex|98.23%|94.01%|**94.65%**|
|Race|91.23%|82.20%|**83.12%**|
|Age (MAE) - Regression|5.44|**6.21**| - |
|Age - Classification|70.1%|-|**67.94%**|

#### Regenerate overall accuracy results
build Sex, Race, Age and Age Classification datasets using UTKFaces. Then execute utkfaces_dataset.py. dataset is the original UTKFACE dataset folder. 
output is the new masked dataset output directory.The types are age, race, sex and age_classification

```sh
cd AJCAI2021/dataset_util
python utkfaces_dataset.py --dataset "./utkfacesdataset_path" --output "../new_masked_utk_face_dataset_path" --type age
```
Download the models to AJCAI2021/dataset_util folder. Then configure the config.json for each model related dataset path, model path and accuracy type (mae or percentage). 
Then execute regenerate_overall_accuracy_results.py as follows.

```sh
cd AJCAI2021
python regenerate_overall_accuracy_results.py
```

### License

MIT

## Citation

If you use the repository for masked face identification, please use the following citation:

@INPROCEEDINGS{seneviratne2021multidataset,
  author={Seneviratne, Sachith and Kasthuriarachchi, Nuran and Rasnayaka, Sanka},
  booktitle={2021 Digital Image Computing: Techniques and Applications (DICTA)}, 
  title={Multi-Dataset Benchmarks for Masked Identification using Contrastive Representation Learning}, 
  year={2021},
  volume={},
  number={},
  pages={01-08},
  doi={10.1109/DICTA52665.2021.9647194}}

If you use the repository for masked face privacy, please use the following citation:

@misc{seneviratne2021does,
      title={Does a Face Mask Protect my Privacy?: Deep Learning to Predict Protected Attributes from Masked Face Images}, 
      author={Sachith Seneviratne and Nuran Kasthuriarachchi and Sanka Rasnayaka and Danula Hettiachchi and Ridwan Shariffdeen},
      year={2021},
      eprint={2112.07879},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

## Acknowledgements

- This project is supported by National Health and Medical Research Grant GA80134. 
- This research was undertaken using the LIEF HPC-GPGPU Facility hosted at the University of Melbourne. This Facility was established with the assistance of LIEF Grant LE170100200. 
- This research was undertaken using University of Melbourne Research Computing facilities established by the Petascale Campus Initiative.




# Master Thesis

This is the repository of my master thesis in KIT.

## TODOs

- [ ] update README
- [ ] upload dataset
- [ ] add Dockerfile

## Platform

All scripts have been tested under Mac OS and Linux.

## Setup

### Manually

All necessary packages can be checked in `requirements.txt`. It's a good practice to use python virtual environment 
to set up the programing environment.

To set up python virtual environment: <br>
Follow the [installation instruction of MiniConda](https://docs.conda.io/en/latest/miniconda.html#) and then run 
the following snippet using your favourite terminal application:
```bash
conda create -n py36 python=3.6
conda activate py36
```

To install all necessary packages:
```bash
cd path_of_project_root
pip install -r requirements.txt
```

### Automatically (using docker)

TBA...

## Dataset

We use [ShapeNetCore](https://shapenet.org/) as our dataset. As for the segmented label, we use 
[shapenetcore_partanno_segmentation_benchmark_v0](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip), 
which is provided by [A Scalable Active Framework
for Region Annotation in 3D Shape Collections](https://cs.stanford.edu/~ericyi/project_page/part_annotation/).

### Prepare dataset from scratch

In order to prepare the dataset from scratch, you need to firstly download 
[ShapeNetVox32](https://cvgl.stanford.edu/data2/ShapeNetVox32.tgz) and 
[shapenetcore_partanno_segmentation_benchmark_v0](https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip). 
ShapeVox32 is the voxelization version of ShapeNetCore, using [binvox](https://www.patrickmin.com/binvox/) for the voxelization. You can also 
voxelize ShapeNetCore using binvox by yourself instead of downloading ShapeVox32. After downloading and unzip those two datasets, run the 
following snippet in your favourite terminal application:
```bash
cd path_of_the_project_root
python utils/data_preprocessing.py path_of_pcd_category path_of_binvox_category output_path
```
For example, `./shapenetcore_partanno_segmentation_benchmark_v0/03001627/` is the pcd path of the category `chair`. 
`./ShapeNetVox32/03001627/` is the binvox path of the category `chair`. Output path is the path where you want to save the 
generated data.

For more usage about `data_preprocessing.py`, please run:
```bash
python utils/data_preprocessing.py -h
```

### Using our dataset

chair: [03001627.zip(681MB)](https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/03001627.zip?inline=false) <br>
number of shapes: 3746 <br>
maximal number of parts: 4

table: [04379243.zip(666MB)](https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/04379243.zip?inline=false) <br>
number of shapes: 5263 <br>
maximal number of parts: 3

airplane: [02691156.zip(278MB)](https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/02691156.zip?inline=false) <br>
number of shape: 2690 <br>
maximal number of parts: 4

lamp: [03636649.zip(200MB)](https://gitlab.com/JunweiZheng93/shapenetsegvox/-/raw/master/03636649.zip?inline=false) <br>
number of shapes: 1546 <br>
maximal number of parts: 4

### Visualize dataset

The generated dataset consists of some PNG images and `.mat` files. To visualize `.mat` files, please run:
```bash
cd path_of_the_project_root
python utils/visualization.py path_of_mat_file
```

## Training

Our training script is user-friendly. The only thing you need to do is to open your favourite terminal application and
type:
```bash
cd path_of_the_project_root
python train.py 
```
Firstly, the training script will ask you if you want to take some notes for the training. If yes, it will open `vim`
and you can start typing. After finish typing, use `:wq` to exit `vim` (check how to use `vim`) and the note will be save automatically. 
The training script will then download the necessary dataset automatically, place the dataset to the correct directory and
start training according to the hyper-parameter you have set. All results (notes you have taken, all hyper-parameters, 
tensorboard logs, model weights in `.h5` format and so on) will be saved automatically in `project_root/results/`.

In order to modify the hyper-parameters for training, please open `hparam.py` and change the value over there.

## Evaluation

### batch mode

The default mode of the evaluation is batch mode. That means it will show you a batch of shapes (default values is 4 
shapes and will be picked randomly in the same category). Firstly, the ground truth shape will be shown for shape #1, and then
the model output shape for shape #1. Secondly, the ground truth shape for shape #2 will be shown and then the model output
shape for shape #2. The process will go on until shape #4.

To run batch mode, please type:
```bash
cd path_of_the_project_root
python evaluate.py model_path --category the_category_you_want_to_choose
```
For example, if you want to evaluate how good you model is for the category `chair` (assume your model is saved in 
`project_root/results/20210723162046/process_3/checkpoint.h5`), you need to type:
```bash
python evaluate.py results/20210723162046/process_3/checkpoint.h5
```
`chair` is the default category. so you don't need to set value for `--category`.

### single mode

Single mode allows you to check the evaluation result for one specific shape. For example, if you want to evaluate how 
good you model is for the shape `03001627/1a38407b3036795d19fb4103277a6b93`(it is a shape in catergory `chair`)
and assume your model is saved in `project_root/results/20210723162046/process_3/checkpoint.h5`, you should type:
```bash
cd path_of_the_project_root
python evaluate.py results/20210723162046/process_3/checkpoint.h5 -m single -s 03001627/1a38407b3036795d19fb4103277a6b93
```

### exchange mode

Exchange mode allows you to exchange parts from different shapes. For example, if you want to change part #1 from shape #1 
and part #1 from shape #2, this is the correct mode for you. Exchange mode will first show the ground truth of the two shape 
you want to exchange, then it will show you the reconstructed shape of the two shape respectively and finally the exchanged 
shape will be shown.

Assume your model is saved in `project_root/results/20210723162046/process_3/checkpoint.h5` and the two shape you want to 
exchange are saved in `03001627/1a38407b3036795d19fb4103277a6b93`, `03001627/1b8e84935fdc3ec82be289de70e8db31`. To run exchange 
mode, just type:
```bash
cd path_of_the_project_root
python evaluate.py results/20210723162046/process_3/checkpoint.h5 -m exchange -e 03001627/1a38407b3036795d19fb4103277a6b93 03001627/1b8e84935fdc3ec82be289de70e8db31 --which_part 1
```

### assembly mode

This mode is used to assemble parts within a batch randomly. It will firstly show the ground truth of every shape and then 
the random assembled shapes.

Assume your model is saved in `project_root/results/20210723162046/process_3/checkpoint.h5` and the shapes used as ground truth 
are saved in `03001627/1a38407b3036795d19fb4103277a6b93`, `03001627/1b8e84935fdc3ec82be289de70e8db31`, and 
`03001627/1f571e244384bca736b0f2a1430e993a`. To run assembly mode, just type:
```bash
cd path_of_the_project_root
python evaluate.py results/20210723162046/process_3/checkpoint.h5 -m assembly -a 03001627/1a38407b3036795d19fb4103277a6b93 03001627/1b8e84935fdc3ec82be289de70e8db31 03001627/1f571e244384bca736b0f2a1430e993a
```

For more usage of `evaluate.py`, please type:
```bash
python evaluate.py -h
```

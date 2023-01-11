# Rock-Cuttings-Characterization

The goal of this project is to guess the rock type from CT-scan cutting with deep learning.

It is based on nfholsen's repository [Rock_Cuttings_Characterisation](https://github.com/nfholsen/Rock_Cuttings_Characterisation) with emphasis on making it easy to use, up to date and running. The PIXE lab at EPFL has the pre-processed images.

The project layout is as follows:

* **config**: where everything relative to the model parameters and dataset combinations is
  * **sets**: **.csv* files containing the paths to the chosen images in the *Data* folder
* **data**: where all images are kept, grouped by origin
  * **Borehole**: in-situ samples
  * **Lab**: In laboratory scanned cuttings
  * **Lab_New**: contains all of Lab images and some new ones
  * **New**: where samples are saved when pre-processing
  * **Raw**: where the raw (not preprocessed yet) images are kept
  * **Train**: A combination of the **Borehole** and **Lab** samples since not every type is available from the **Lab** set
* **outputs**: model results will be saved here, sorted by model name defined in the config (\**.yaml*)) files
* report: image examples and plots
* **scripts**
  * **Utils**: modules to lighten the main scripts. *Note: the path_to function is particularily useful as it takes a relative path with the project as root and transforms it into an absolute path. All paths in the codes should be relative or used via path_to()*
  * \**.py* executables and modules
  * *main.ipynb* for any purposes
  * *scripts.log* to help understand and debug the project

## Step-by-step usage: suppose we have a new set of untreated new images

1. Pre-process the files using the ***preprocess.py*** script. You can either call its function *preprocess(path_to_raw_folders, folder_to_save_in)* or run the script:
   *python3 scripts/preprocess.py -i data/Raw -o data/New*
2. Write a \**.csv* file containing the image paths and labels or multiple files if you want to split them into train and validations sets already. The script *make_sets.py* I used could be useful. It should be saved in *config/sets*
3. Write the \**.yaml* config file, you can use the *config/debugconfig.yaml* file as a template
4. *Train a model by calling the *train()* function in ***scripts/train.py*** or run the script:*

   *python3 scripts/train.py config/config_file.yaml*

   The outputs with a results log should appear under **outputs/model_name**
5. *Test your model with ***scripts/test.py****

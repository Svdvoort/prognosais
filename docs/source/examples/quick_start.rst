Quick start
=================

This page will show a quick-start of using prognosais, including installation and a simple toy experiment.

Installation
--------------

Prognosais can be installed using ``pip``. It is recommended to install prognosais in a virtual environment:
The following code block creates a virtual environment and installs prognosais in a linux environment.

.. code-block:: bash

  mkdir ~/prognosais && cd ~/prognosais
  python3 -m venv env
  source env/bin/activate
  pip install prognosais

Example experiment
-------------------

We will now set up an example experiment to show how prognosais works and to explain the settings of prognosais.
First we set up the experiment by installing the additionaly required packages and obtain the code of the examples:
(This assumes that the virtual environment has been set-up as specified under installation)

.. code-block:: bash

  cd ~/prognosais
  source env/bin/activate
  pip install xlrd
  git clone https://github.com/Svdvoort/prognosais_examples
  cd prognosais_examples

Now, we need to download the data for the example experiment.
This data is part of the `'LGG-1p19qDeletion' collection on TCIA <https://doi.org/10.7937/K9/TCIA.2017.dwehtz9v>`_
More information can be found in the `accompanying publication <https://doi.org/10.1007/s10278-017-9984-3>`_.

We will now download the data to a directory of our choosing:

.. code-block:: bash

  cd prognosais_examples
  python download_1p19q_data.py

You will now be prompted for a directory in which to save the data.
Wait untill the data is done downloading and extracting.
The script will have prepared the input data and the download data, you can have a look in the download folder you specified.

The script will provide the input folder and label file that need to be specified.
Open the ``config.yml`` file (in the prognosais_examples folder), you can have a look here at the different settings, which are explained more in depth in the file itself.
For now we need to change three parameters:

input_folder under general, which is set to ``/path/to/input/``, needs to be changed to the input folder provided by the download script
label_file under preprocessing > labeling which is set to ``\path\to\label_file``, needs to be changed to the label file provided by the download script
output_folder under general, which is set to ``/path/to/output``, needs to be changed to a folder of your choice in which the output will be saved.
If you want to speed-up the pre-processing you can also change the 'max_cpus' setting in preprocessing > general.
By default, this is set to 1 which means that only 1 cpu core will be used, increase this if you have multiple cores available.

Once this is done the experiment can simply be run with

.. code-block:: bash

  python example_pipeline.py

This will run the pipeline, including the pre-processing of the scans, the training of the model (a ResNet) and the evaluation of the model on the validation and test set.
The results will be placed in the folder you specified under 'output_folder', in a subfolder starting with ResNet_18.
This folder contains the pre-processed samples, the trained model (including logs from callbacks), and the evaluated results.




# Domain Adpatation For Electromagnetic Imaging.

This repository contains the support code for building complex domain adaptation network for electromagentic imaging proposed in:

```
filler.
```

## Installation

### Requirements

The model proposed is implemented in Pytorch as uses serveral other libraries.

* `torch==1.6`
* `numpy==1.16`
* `matplotlib==1.5.1`
* `scipy==1.2.1`

Setup

To install the package, simply clone the repository:

```
git clone https://github.com/kernelized/emi_da;
cd emi_da;
```

## Running the Model


# The repository is structured as follows:

* `lib.py` has the helper functions for building the model.
* `toolbox.py` has a large accessory functions for various purposes.
* `.py` has the model.


## DATA

Data and Final saved models can be found seprarately outside this repository [here](https://drive.google.com/drive/folders/11QoRjUuBUjZLF9eTdL0cz9PmasKIxv-w?usp=sharing)

* Those folders must be downloaded and added to the repository.
* `data` folder contains the processed data from source domain and target domain. One can experiment with DA seprately with this data. 
    * All details of hardware used to collect target data are abstracted away.
    * All details of simulations used to collect source data are abstracted away.
* `ResultModels` folder contain saved models of DANN.


# Feedback

Send feedback to [Alex Al-Saffar](a.alsaffar@uqconnect.edu.au).

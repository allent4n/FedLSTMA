# FedLSTMA
## This is an official implementation of the paper: ***Privacy-Preserving Federated Learning for Proactive Maintenance of IoT-Empowered Multi-Location Smart City Facilities***. 
### In this Repo, we aim at filling the gap of privacy-preserving federated learning framework by providing a federated LSTM Autoencoder model (FedLSTMA) for proactive maintenance of public smart toilets (i.e., public toilets equipped with IoT sensors). 
### Thanks to [Pyfhel](https://github.com/ibarrond/Pyfhel), we are able to provide an effective and efficient way to do model update encryption based on the FHE manner.

## Framework
### ***Overall framework (FHE-based)***
<img width="1717" alt="sys_framework_v2" src="https://github.com/allent4n/federated-learning-with-fully-homomorphic-encryption/assets/78404109/cd68c6d8-9242-4c92-aa51-a5c44b097a21">

### ***Experiment*** (Connectivity of server and clients based on IP address and port number)
<img width="464" alt="simulation" src="https://github.com/allent4n/federated-learning-with-fully-homomorphic-encryption/assets/78404109/9e907cdf-c530-42ef-9aed-f2111f64d265">

## Model Structure (LSTM-based Autoencoder)
### LSTM-based Autoencoder
<img width="680" alt="image" src="https://github.com/allent4n/federated-learning-with-fully-homomorphic-encryption/assets/78404109/c7d0cb4f-a28d-4551-981f-e9fb5ff7ce99">

1. Generate input-like output
2. If the loss is larger than a threshold, then consider the device is in abnormal status.

## Results
<img width="404" alt="image" src="https://github.com/allent4n/federated-learning-with-fully-homomorphic-encryption/assets/78404109/0ac66ee9-7824-4131-923a-b451bffeb538">



Performance of Global Model of FedLSTMA and Traditional Centralized Machine Learning Approach (Baseline) under Four Different Local Sites


## Clone this Repo
Step 1: Go to [Pyfhel](https://github.com/ibarrond/Pyfhel), follow the steps to clone and install this magnificant repo first! 🥰

Step 2: Clone this repo
* Change the directory to the installed Pyfhel folder.
```
cd Pyfhel
```

* Clone this repo inside the Pyfhel folder with the following code
```
git clone https://github.com/allent4n/federated-learning-with-fully-homomorphic-encryption
```

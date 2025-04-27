# FedLSTMA
## This is an implementation of the paper: ***[Privacy-Preserving Federated Learning for Proactive Maintenance of IoT-Empowered Multi-Location Smart City Facilities](https://www.sciencedirect.com/science/article/abs/pii/S1084804524001735)***. 
### In this Repo, we aim at filling the gap of privacy-preserving federated learning framework by providing a federated LSTM Autoencoder model (FedLSTMA) for proactive maintenance of public smart toilets (i.e., public toilets equipped with IoT sensors). 
### Thanks to [Pyfhel](https://github.com/ibarrond/Pyfhel), we are able to provide an effective and efficient way to do model update encryption based on the FHE manner.

## Framework
### ***Overall framework (FHE-based)***
<img width="639" alt="image" src="https://github.com/user-attachments/assets/f14417c4-90c3-4ef6-b91f-2869327a90d7" />

## Model Structure (LSTM-based Autoencoder)
### LSTM-based Autoencoder
<img width="341" alt="image" src="https://github.com/user-attachments/assets/f8a679f3-51f4-4728-b141-3223e8981b99" />


## Results
<img width="404" alt="image" src="https://github.com/allent4n/federated-learning-with-fully-homomorphic-encryption/assets/78404109/0ac66ee9-7824-4131-923a-b451bffeb538">

Detailed results can be found in the [paper](https://www.sciencedirect.com/science/article/abs/pii/S1084804524001735).


## QuickStart
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

* Run the code (Note: ***this code will initalize the real network connections among different cites***)

```
python <cite>/main.py         # run main.py from each sub-folder, cite can be female/male/office/shop and server
```

## 🔎 Citation

```
@article{tan2024privacy,
  title={Privacy-preserving federated learning for proactive maintenance of IoT-empowered multi-location smart city facilities},
  author={Tan, Zu-Sheng and See-To, Eric WK and Lee, Kwan-Yeung and Dai, Hong-Ning and Wong, Man-Leung},
  journal={Journal of Network and Computer Applications},
  volume={231},
  pages={103996},
  year={2024},
  publisher={Elsevier}
}
```

# Offtarget_drugsafety
code for "In Silico Off-Target Profiling for Enhanced Drug Safety Assessment"

### Abstract

Ensuring drug safety in the early stages of drug development is crucial to avoid costly failures in subsequent phases. However, the economic burden associated with detecting drug off-targets and potential side effects through in vitro safety screening and animal testing is substantial. Drug off-target interactions, along with the adverse drug reactions they induce, are significant factors affecting drug safety. To assess the liability of candidate drugs, we developed an artificial intelligence model for the precise prediction of compound off-target interactions, leveraging multi-task graph neural networks. The outcomes of off-target predictions can serve as representations for compounds, enabling the differentiation of drugs under various ATC codes and the classification of compound toxicity. Furthermore, the predicted off-target profiles are employed in ADR enrichment analysis, facilitating the inference of potential ADRs for a drug. Using the withdrawn drug Pergolide as an example, we elucidate the mechanisms underlying ADRs at the target level, contributing to the exploration of the potential clinical relevance of newly predicted off-target interactions. Overall, our work facilitates the early assessment of compound safety/toxicity based on off-target identification, deduces potential ADRs of drugs, and ultimately promotes the secure development of drugs.
<img src="https://github.com/myzhengSIMM/Offtarget_drugsafety/blob/main/databases/TOC.png" alt="Offtarget Workflow">

Below are the instructions for training models, predicting off-target profiles, applying these predictions, and setting up the required environment.

### 1.Train off-target profile prediction model

To train the off-target profile prediction model, execute the following script:
```
bash ./drugsafety/train.sh
```
This script retrains the model for seven types of targets. Here we used the kinases dataset to demonstrate the training process of our model. The trained model parameters are is now available freely at https://drive.google.com/drive/folders/14eQVdXwSeLXOPG2lG06509Ory7had2tN?usp=sharing, user can download directly into the './drugsafety/results' folder.

### 2.Predict off-target profile of compounds

To use the trained model for predicting off-target profiles of compounds, run:
```
python ./drugsafety/predict.py
```
The result will be save in './drugsafety/predict/predict_result' folder.


### 3.Apply the off-target profile prediction results
The predicted off-target profiles can be employed as molecular representations for the subsequent classification of a drug's ATC, toxicity, as well as ADR enrichment analysis.

#### 3.1 ATC classification
We processed the collected ATC data into multi-label format in the './offtarget_application/ATC_classification/data_pro.ipynb' file, took the off-target predicted results of ATC related compounds as compound features, and ran the following code to train the ATC classification model.
```
python ./offtarget_application/ATC_classification/model/mlknn_offtarget.py
```

#### 3.2 Toxicity prediction
We processed the toxic-related data in the './offtarget_application/Toxicity_prediction/toxic_data_pro.ipynb' file, took the off-target predicted results as compound features, and ran the '.ipynb' files under the  './offtarget_application/Toxicity_prediction/model ' folder  to train the different toxicity prediction model.

#### 3.3 ADR enrichment analysis
According to the off-target panel prediction result of a compound, we obtain the interaction off-targets (gene_list), input the data into './offtarget_application/ADR_enrichment_analysis/enrich_code_drugs.ipynb' file, and run the file to obtain the ADR enrichment analysis results of the compound.

### 4.Setup and dependencies
The project environment requirements are listed in 'requirements.txt'.

### 5.Requirements
python = 3.8.0  
pytorch = 1.11.0  
deepchem = 2.6.1  
dgllife = 0.2.9  
scikit-learn = 0.24.2  
numpy = 1.22.1  
pandas = 1.4.2  
rdkit = 2022.03.2  
gseapy = 1.1.0  

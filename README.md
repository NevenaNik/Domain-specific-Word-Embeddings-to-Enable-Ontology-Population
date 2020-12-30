# Domain-specific Word Embeddings to Enable Ontology Population

Our project is structured into three parts:  


* Data Preprocessing
* Model Training 
* Model Evaluation

Note:  
Model training is split into phases 1 to 5.  
For Phase 0 models trained in phase 2 are evaluated.  
For Phase 6 models trained in phase 3 are evaluated.  
  

################################CODE####

### Preprocessing
For **preprocessing** use: `preprocessing.py`
  

### Model Training and Evaluation (Phase 0 to Phase 4)
For **model training** use: `training.py`  
For **model evaluation approach1** use: `evaluation_approach1.py`  
To do **model training** and **evaluation (approach 1)** at once adjust: `run.py`  
  

### Model Training and Evaluation (Phase 5)
For **model training** use: `notebooks/training/modeltraining_phase5.py`  
For **model evaluation approach1** use: `evaluation_approach1.py`
  

### Model Evaluation (Phase 6)
For **model evaluation approach2** use: `evaluation_approach2.py`
  
  
################################DATA####

### Terminology
All preprocessing adjustments are stored in folder `terminology/`, included are:  
* custom stopwords
* abbreviations + long-forms
* considered n-grams
* normalization steps
  
  
### Evaluation Data
All evaluation data sets are stored in folder `evaluation/`.  
Expert data is stored in `evaluation/expert`, divided by key terms (see `evaluation/eval_approach_1_key.json`).  
  

### Models
All information on model training are stored in folder `models/`, divided by experimental phase.  
  
  
### Results
All evalaution results are stored in folder `results/`, divided by experimental phase.
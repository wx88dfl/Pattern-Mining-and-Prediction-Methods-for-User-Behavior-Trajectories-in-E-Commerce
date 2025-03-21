# Pattern-Mining-and-Prediction-Methods-for-User-Behavior-Trajectories-in-E-Commerce
Installation
Ensure you have the required dependencies installed:

bash
pip install numpy pandas scikit-learn hmmlearn fastdtw hdbscan mlxtend torch transformers
Usage
1. Run the Main Program
bash

python main.py
2. Project Structure
less
                   
│   ├── preprocess.py           # Data preprocessing module
│   ├── clustering.py           # Trajectory clustering module
│   ├── prediction.py           # Trajectory prediction module 
│   ├── evaluation.py           # Evaluation module (P@1, P@6, MRR)
│   ├── frequent_patterns.py    # Frequent trajectory mining module
│   ├── main.py                 # Main entry file
└── README.md    

# Two-Stream-Framework-for-Dynamic-Graphs
Welcome to the GitHub repository for my research thesis titled 'A GNN-based Two-Stream Framework for Dynamic Link Prediction.' I undertook this research during my MSc in Machine Learning & Data Science at Imperial College London.

## Repository Structure
Below are descriptions of the main folders in this project:

| Folder Name | Description |
|-------------|-------------|
| `config` | YAML configuration files for the models explored in the research paper |
| `data` | Placeholder for raw data (currently empty). [Download raw data here](https://drive.google.com/file/d/1_vP6kPl_Sg1_I8JlLtk3WtSaHEMJenfU/view?usp=drive_link) |
| `model` | Placeholder for trained models' checkpoints (currently empty). [Download them here](https://drive.google.com/file/d/1_vP6kPl_Sg1_I8JlLtk3WtSaHEMJenfU/view?usp=drive_link) |
| `prep_data` | Processed data using the `link_prediction_tasker` object (currently empty). [Download here](https://drive.google.com/file/d/1igNNj-REZ4gxomnfvfiGQqZeObF2NVVM/view?usp=sharing) |
| `script` | Jupyter notebook script history of training/validating models, sourced from Google Colab |

For checking the model's performance, refer to the corresponding scripts in the `script` folder. To re-run the experiment, download the necessary content for `data`, `model`, and `prep_data` folders.

## Raw Data Sources
You can download the raw datasets from the following links. Refer to `run_as.py` for examples on data loading, preparation, and model training/validation.

- **Bitcoin OTC**: [Download here](http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html)
- **Bitcoin Alpha**: [Download here](http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html)
- **UC Irvine**: [Download here](http://konect.uni-koblenz.de/networks/opsahl-ucsocial)
- **Autonomous Systems**: [Download here](http://snap.stanford.edu/data/as-733.html)

## Scripts Description
Below are descriptions of the main scripts in this project:

| Script | Description |
|--------|-------------|
| `custom_loss` | Holds the custom weighted loss object for model training |
| `data_auto_sys` | Data loader for the autonomous system dataset |
| `data_bitcoin` | Data loader for the Bitcoin OTC and Bitcoin Alpha datasets |
| `data_UCI` | Data loader for the UC Irvine dataset |
| `graph_similarity` | Function for calculating Jaccard similarity between consecutive graphs |
| `models` | Contains all model objects |
| `run_as` | Example script to run experiments on the autonomous system dataset |
| `splitter` | Contains the splitter object for splitting processed data into train/validation/test sets |
| `tasker_link_prediction` | Contains the tasker object to process raw data (e.g., calculating initial node features) |
| `trainer` | Trainer object for model training and validation |
| `util` | General utility functions |
| `util_tasker` | Utility functions related to `tasker_link_prediction` |

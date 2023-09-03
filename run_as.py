import data_auto_sys as aus
import tasker_link_prediction as t_lp
from splitter import splitter
from models import TwoStream_GCN
from trainer import Trainer
import yaml
import os

# if the Link_Pred_Tasker need to prepare the data
prep = False

with open('config/config_AS.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_path = config['model_path']

if not os.path.exists(model_path):
    os.mkdir(model_path)
    os.mkdir(model_path + "best_checkpoints/")
    os.mkdir(model_path + "latest_checkpoints/")

with open(model_path + 'config.yaml', 'w') as file:
    yaml.dump(config, file)

print(config)

data = aus.Autonomous_System_Dataset(config['data_path'])
tasker = t_lp.Link_Pred_Tasker(data, path=config['prep_data_path'], prep=prep, 
                               embs_dim=config['spatial_input_dim'], temp_dim=int(config['temporal_input_dim']/10),
                               major_threshold=config['major_threshold'], smart_neg_sampling=True, neg_sample=1000)

splitter_as = splitter(tasker, train_proportion = config['train_proportion'], dev_proportion = config['dev_proportion'])

model= TwoStream_GCN(spatial_input_dim=config['spatial_input_dim'],
                 temporal_input_dim=config['temporal_input_dim'],
                 spatial_hidden_size=config['spatial_hidden_size'],
                 temporal_hidden_size=config['temporal_hidden_size'],
                 classifier_hidden_size=config['classifier_hidden_size'],
                 gcn_fusion_size=config['gcn_fusion_size'],
                 ffn_fusion_size=config['ffn_fusion_size'],
                 ffn_hiden_size=config['ffn_hidden_size'])

trainer = Trainer(model=model, splitter=splitter_as, model_path=model_path, adam_config=config['adam_config'], patience=config['patience'])


# trainer.train(config['num_epochs'])
# trainer.resume_training(1000)
trainer.restore_best_checkpoint()
trainer.validate('Validation')
trainer.validate('Test')
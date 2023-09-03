import numpy as np
import tensorflow as tf
import util as u
from custom_loss import cross_entropy
import util_tasker as ut


class Trainer:
    def __init__(self, model, splitter, model_path, adam_config=None, patience=3, **kwargs):
        """
        Initialize the Trainer object.

        :param model: Model to be trained.
        :param splitter: Object containing split datasets (train, dev, test).
        :param adam_config: Optional configuration dictionary for the Adam optimizer.
        """
        
        self.model = model
        if 'class_weight' in kwargs.keys():
            self.loss_fn = cross_entropy(kwargs['class_weight']) 
        else:
            self.loss_fn = cross_entropy()  # Custom loss function for training
        
        # Setting up the optimizer based on provided configurations or defaults.
        if adam_config:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=adam_config['adam_learning_rate'],
                                                    beta_1=adam_config['adam_beta_1'],
                                                    beta_2=adam_config['adam_beta_2'],
                                                    epsilon=adam_config['adam_epsilon'])
        else:
            self.optimizer = tf.keras.optimizers.Adam()

        # Lists to store evaluation metrics values for each batch during training/validation
        self.MRRs = []
        self.MAPs = []
        self.batch_sizes = []
        
        # History lists to store metric values across epochs
        self.train_MRR_hist = []
        self.dev_MRR_hist = []

        self.train_MAP_hist = []
        self.dev_MAP_hist = []

        # Split datasets
        self.train_data = splitter.train
        self.dev_data = splitter.dev
        self.test_data = splitter.test

        # Early stopping attributes
        self.patience = patience
        self.best_val_metric = float('-inf')  # Assuming higher MRR is better, change if otherwise
        self.wait = 0

        # Model checkpoint paths
        self.model_path = model_path
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.best_checkpoint_prefix = self.model_path + "best_checkpoints/"
        self.latest_checkpoint_prefix = self.model_path + "latest_checkpoints/"

        self.epoch=0

    def train_step(self, data):
        """
        Execute a single training step: forward pass, loss computation, backward pass, and optimization.

        :param data: Input batch data.
        :return: Computed loss for the current batch.
        """
        
        with tf.GradientTape() as tape:
            logits = self.model(data, training=True)
            labels = data['label_sp']._values
            loss = self.loss_fn(labels, logits)
            
        # Compute gradients and optimize model parameters
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Calculate evaluation metrics for the current batch
        MRR = u.get_MRR(labels, logits, data['label_sp']._indices)
        MAP = u.get_MAP(labels, logits)
        
        # Append metrics values to the respective lists
        self.MRRs.append(MRR)
        self.MAPs.append(MAP)
        self.batch_sizes.append(len(labels))

        return loss

    def train(self, num_epochs):
        """
        Train the model for a specified number of epochs.

        :param num_epochs: Number of epochs to train.
        """
        
        for epoch in range(num_epochs):
            # Reset metrics lists for the new epoch
            self.MRRs = []
            self.MAPs = []
            self.batch_sizes = []

            for step, data in enumerate(self.train_data):
                loss = self.train_step(data)
                print(f"Epoch {epoch+1+self.epoch}, Step {step+1}, Loss: {loss}")

            # Calculate average metrics values for the epoch
            epoch_train_MRR = self.calc_epoch_metric(self.MRRs, self.batch_sizes)
            epoch_train_MAP = self.calc_epoch_metric(self.MAPs, self.batch_sizes)

            print(f'Train Metric MRRs: {epoch_train_MRR}')
            print(f'Train Metric MAPs: {epoch_train_MAP}')

            self.train_MRR_hist.append(epoch_train_MRR)
            self.train_MAP_hist.append(epoch_train_MAP)

            # Validate on dev and test sets
            epoch_dev_MRR, epoch_dev_MAP = self.validate('Validation')
            self.dev_MRR_hist.append(epoch_dev_MRR)
            self.dev_MAP_hist.append(epoch_dev_MAP)

            # Save the latest model (including optimizer state)
            self.save_latest_checkpoint()
               

            if len(self.dev_MRR_hist) > 0 and self.dev_MRR_hist[-1] == max(self.dev_MRR_hist):
                self.save_best_checkpoint()
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print("Early stopping triggered!")
                    break

    def calc_epoch_metric(self, metric_val, batch_sizes):
        """
        Compute weighted average of a metric over an epoch.

        :param metric_val: List of metric values for each batch.
        :param batch_sizes: List of sizes for each batch.
        :return: Weighted average metric value for the epoch.
        """
        
        metric_val, batch_sizes = np.array(metric_val), np.array(batch_sizes)
        return sum(metric_val * batch_sizes) / sum(batch_sizes)
    
    def validation_step(self, data):
        """
        Evaluate the model on a given batch without updating its parameters.

        :param data: Input batch data.
        :return: Computed loss for the current batch.
        """
        
        logits = self.model(data, training=False)
        labels = data['label_sp']._values
        loss = self.loss_fn(labels, logits)
        
        # Calculate evaluation metrics for the batch
        MRR = u.get_MRR(labels, logits, data['label_sp']._indices)
        MAP = u.get_MAP(labels, logits)
        
        # Append metrics values to the respective lists
        self.MRRs.append(MRR)
        self.MAPs.append(MAP)
        self.batch_sizes.append(len(labels))

        return loss

    def validate(self, mode):
        """
        Evaluate the model's performance on either the dev or test set.

        :param mode: 'Validation' for the dev set and 'Test' for the test set.
        :return: Average MRR and MAP values for the specified dataset.
        """
        
        # Reset metrics lists for the new validation/test
        self.MRRs = []
        self.MAPs = []
        self.batch_sizes = []

        if mode == 'Validation':
            for data in self.dev_data:
                self.validation_step(data)
        
        if mode == 'Test':
            for data in self.test_data:
                self.validation_step(data)

        epoch_MRR = self.calc_epoch_metric(self.MRRs, self.batch_sizes)
        epoch_MAP = self.calc_epoch_metric(self.MAPs, self.batch_sizes)

        print(f'{mode} Metric MRRs: {epoch_MRR}')
        print(f'{mode} Metric MAPs: {epoch_MAP}')
        
        return epoch_MRR, epoch_MAP
    
    def save_best_checkpoint(self):
        self.checkpoint.save(file_prefix=self.best_checkpoint_prefix)

    def save_latest_checkpoint(self):
        self.checkpoint.save(file_prefix=self.latest_checkpoint_prefix)
        ut.save_pkl(self.train_MRR_hist, self.latest_checkpoint_prefix + 'train_MRR')
        ut.save_pkl(self.train_MAP_hist, self.latest_checkpoint_prefix + 'train_MAP')
        ut.save_pkl(self.dev_MRR_hist, self.latest_checkpoint_prefix + 'dev_MRR')
        ut.save_pkl(self.dev_MAP_hist, self.latest_checkpoint_prefix + 'dev_MAP')


    def restore_best_checkpoint(self):
        """
        Restore the best checkpoint. 
        This can be used to resume training with the best model or for evaluation.
        """
        self.checkpoint.restore(tf.train.latest_checkpoint(self.best_checkpoint_prefix))


    def restore_latest_checkpoint(self):
        """
        Restore the latest checkpoint. 
        This can be used to resume training with the best model or for evaluation.
        """
        self.checkpoint.restore(tf.train.latest_checkpoint(self.latest_checkpoint_prefix))
        self.train_MRR_hist = ut.load_pkl(self.latest_checkpoint_prefix + 'train_MRR')
        self.dev_MRR_hist = ut.load_pkl(self.latest_checkpoint_prefix + 'dev_MRR')


        self.train_MAP_hist = ut.load_pkl(self.latest_checkpoint_prefix + 'train_MAP')
        self.dev_MAP_hist = ut.load_pkl(self.latest_checkpoint_prefix + 'dev_MAP')

        self.epoch=len(self.dev_MRR_hist)


    def resume_training(self, num_epochs):
        # Load the latest model (including optimizer state)
        self.restore_latest_checkpoint()
        # Retrieve the last epoch from the saved history (assuming you want to continue for the same number of epochs)
        last_epoch = len(self.train_MRR_hist)
        # Resume training
        self.train(num_epochs=num_epochs)
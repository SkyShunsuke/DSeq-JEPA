import wandb
import os
import csv
import tempfile
import shutil

class CSVLogger:
    def __init__(self, log_dir, rank, filename="log.csv"):
        """
        CSV logger that automatically adds new columns when new metric keys appear.
        Keeps existing data intact.
        Only rank 0 will write to the CSV file.
        """
        self.rank = rank
        self.use_csv = (rank == 0)
        self.fieldnames = None
        self.writer = None

        if self.use_csv:
            self.log_file = os.path.join(log_dir, filename)
            # Create file if it does not exist
            file_exists = os.path.exists(self.log_file)
            self.csv_file = open(self.log_file, mode='a+', newline='')
            self.csv_file.seek(0)

            # Load existing header if present
            reader = csv.reader(self.csv_file)
            try:
                header = next(reader)
                self.fieldnames = header
            except StopIteration:
                self.fieldnames = None

    def log_metrics(self, metrics, step=None):
        """Log metrics to CSV file. Adds new columns dynamically if needed."""
        if not self.use_csv:
            return

        # Build the row dictionary
        row = {'step': step} if step is not None else {}
        row.update(metrics)

        # If no writer yet or new keys appear, rebuild the CSV file
        new_fields = set(row.keys()) - set(self.fieldnames or [])
        if self.writer is None or new_fields:
            # Update fieldnames
            self.fieldnames = sorted(set(self.fieldnames or []).union(row.keys()))

            # Rebuild file while keeping previous data
            tmp_file = tempfile.NamedTemporaryFile('w', delete=False, newline='')
            writer = csv.DictWriter(tmp_file, fieldnames=self.fieldnames)
            writer.writeheader()

            # Copy old data if any
            self.csv_file.seek(0)
            reader = csv.DictReader(self.csv_file)
            for old_row in reader:
                writer.writerow(old_row)
            
            tmp_file.flush()
            tmp_file.close()

            # Replace old file with new one
            self.csv_file.close()
            shutil.move(tmp_file.name, self.log_file)
            self.csv_file = open(self.log_file, mode='a', newline='')
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        else:
            # Normal case: writer already initialized
            if self.writer is None:
                self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
                if self.csv_file.tell() == 0:
                    self.writer.writeheader()

        # Write the new row
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        """Close the CSV file safely."""
        if self.use_csv and self.csv_file:
            self.csv_file.close()

class TensorboardLogger:
    def __init__(self, log_dir, rank):
        """Initialize TensorboardLogger. Only the main process (rank 0) will log to TensorBoard.
        param: log_dir: Directory to save TensorBoard logs.
        param: rank: Process rank.
        """
        self.rank = rank
        self.use_tensorboard = (rank == 0)
        if self.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
            
    def log_metrics(self, metrics, step=None):
        """Log metrics to TensorBoard if this is the main process.
        param: metrics: Dictionary of metrics to log.
        param: step: Optional step number.
        """
        if self.use_tensorboard:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, global_step=step)
    
    def close(self):
        """Close the TensorBoard writer if this is the main process."""
        if self.use_tensorboard:
            self.writer.close()

class WandbLogger:
    def __init__(self, project_name, run_name, entity, config, rank):
        """Initialize WandbLogger. Only the main process (rank 0) will log to Weights & Biases.
        param: project_name: Name of the W&B project.
        param: run_name: Name of the W&B run.
        param: entity: W&B entity (username or team name).
        param: config: Configuration dictionary.
        param: rank: Process rank.
        """
        self.rank = rank
        self.use_wandb = (rank == 0)
        if self.use_wandb:
            import wandb
            wandb.init(project=project_name,
                       name=run_name,
                       entity=entity,
                       config=config)
            self.wandb = wandb
        else:
            self.wandb = None
            
    def log_metrics(self, metrics, step=None):
        """Log metrics to Weights & Biases if this is the main process.
        param: metrics: Dictionary of metrics to log.
        param: step: Optional step number.
        """
        if self.use_wandb:
            self.wandb.log(metrics, step=step)
    
    def close(self):
        """Finish the W&B run if this is the main process."""
        if self.use_wandb:
            self.wandb.finish()
    
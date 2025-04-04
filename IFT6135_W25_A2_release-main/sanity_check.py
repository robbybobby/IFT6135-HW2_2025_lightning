import os
import torch
import numpy as np
import pandas as pd
from train import Arguments, train_m_models
from checkpointing import get_extrema_performance_steps_per_trials
from plotter import plot_loss_accs



# Create directory if it doesn't exist
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

# Save loss/accuracy plots for one run
def save_plot(metrics, model_name, plot_path):
    plot_loss_accs(
        metrics,
        multiple_runs=True,
        log_x=False,
        log_y=False,
        fileName="loss_accuracy_plot",
        filePath=plot_path,
        show=True,
    )
    return plot_path

# Save summary table for one model across r_train values
def save_summary_table(summary_dict, model_name, table_latex_caption, table_latex_filename, save_path):
    rows = []
    for r, m in summary_dict.items():
        rows.append([
            r,
            f"{m['min_train_loss']:.4f} ± {m['min_train_loss_std']:.4f}",
            f"{m['min_test_loss']:.4f} ± {m['min_test_loss_std']:.4f}",
            f"{m['max_train_accuracy']:.4f} ± {m['max_train_accuracy_std']:.4f}",
            f"{m['max_test_accuracy']:.4f} ± {m['max_test_accuracy_std']:.4f}",
            f"{m['min_train_loss_step']:.0f} ± {m['min_train_loss_step_std']:.0f}",
            f"{m['min_test_loss_step']:.0f} ± {m['min_test_loss_step_std']:.0f}",
            f"{m['max_train_accuracy_step']:.0f} ± {m['max_train_accuracy_step_std']:.0f}",
            f"{m['max_test_accuracy_step']:.0f} ± {m['max_test_accuracy_step_std']:.0f}",
            f"{m['max_test_accuracy_step'] - m['max_train_accuracy_step']:.0f}",
            f"{m['min_test_loss_step'] - m['min_train_loss_step']:.0f}",
        ])
    df = pd.DataFrame(rows, columns=[
        "r_train", "Min Train Loss", "Min Val Loss", "Max Train Acc", "Max Val Acc",
        "Step Min Train Loss", "Step Min Val Loss", "Step Max Train Acc", "Step Max Val Acc",
        "Δt(Acc)", "Δt(Loss)"
    ])
    latex_table = df.to_latex(index=False, escape=False,
                               caption=table_latex_caption,
                               label=f"tab:{table_latex_caption}")
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{table_latex_filename}.tex"), "w") as f:
        f.write(latex_table)


def sanity_check_experiment(models, r_train_values, gdrive_dir):
  
    summary_per_model = {}

    for model_name in models:
        summaries = {}
        for r in r_train_values:
            args = Arguments()
            args.model = model_name
            args.r_train = r
            args.n_steps = 10**4 + 1
            args.eval_first = 100
            args.eval_period = 100
            # args.seed = 0
            experiment_dir = f"{gdrive_dir}/sanity_check_logs_update/{model_name}"
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            base_path = experiment_dir

            print(f"\n==> Running {model_name.upper()} with r_train={r}")
            _, all_metrics, _ = train_m_models(args, experiment_dir, M=2, seeds=[0, 42])
            
            # Save summary per r_train
            summaries[r] = get_extrema_performance_steps_per_trials(all_metrics)

            # Save plots
            plot_path = ensure_dir(os.path.join(base_path, "plots"))
            save_plot(all_metrics, model_name, plot_path=plot_path)
      
        # Save summaries of r_trains per model
        summary_per_model[model_name] = summaries

        # Save LaTex table per model
        table_path = ensure_dir(os.path.join(base_path, "latex_tables"))
        save_summary_table(summaries, model_name, "Sanity_Check_LSTM_vs_GPT", "sanity_check_table", table_path)

    return summary_per_model



r_train_values = [.5]
models = ["lstm", "gpt"]
log_drive_dir = "./IFT6135_HW2_logs"

sanity_check_summary = sanity_check_experiment(models, r_train_values, log_drive_dir)
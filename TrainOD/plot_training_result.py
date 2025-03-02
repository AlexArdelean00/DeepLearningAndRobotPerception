import matplotlib.pyplot as plt
import pandas as pd

def main(training_results_file):
    # Read data from CSV
    df = pd.read_csv(training_results_file, header=None)

    labels = ['Loss', 'Train Accuracy', 'Test Accuracy', 'Train IoU', 'Test IoU']

    # Plot all columns on the same plot with different colors
    plt.figure()
    for i, col in enumerate(df.columns):
        plt.plot(df.index, df[col], label=labels[i])

    plt.title('Training')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("training_results_file", help="training results file")
    args = parser.parse_args()
    main(args.training_results_file)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_train_val():
    # Import dataframe
    df = pd.read_csv("/home/laptopmindee/Documents/exp_batch.csv")

    # Split and rename dataframes
    df_0 = df[["step", "0-train", "0-val"]]
    df_0 = df_0.rename(columns={'0-train': 'Train', '0-val': 'Val'})
    df_1 = df[["step", "1-train", "1-val"]]
    df_1 = df_1.rename(columns={'1-train': 'Train', '1-val': 'Val'})
    df_3 = df[["step", "3-train", "3-val"]]
    df_3 = df_3.rename(columns={'3-train': 'Train', '3-val': 'Val'})

    dfm_0 = df_0.melt('step', var_name='exp', value_name='Accuracy')
    dfm_1 = df_1.melt('step', var_name='exp', value_name='Accuracy')
    dfm_3 = df_3.melt('step', var_name='exp', value_name='Accuracy')

    # Configure plot
    sns.set_theme()
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    fig.suptitle("Train & Val accuracy on 10 epochs (MNIST) with modes 0, 1 and 3")
    axes[0].set_title("mode 0: without BN")
    axes[1].set_title("mode 1: with basic BN")
    axes[2].set_title("mode 3: with smart BN")

    # Plot
    sns.lineplot(data=dfm_0, x="step", y="Accuracy", hue="exp", palette="viridis", ax=axes[0])
    sns.lineplot(data=dfm_1, x="step", y="Accuracy", hue="exp", palette="viridis", ax=axes[1])
    sns.lineplot(data=dfm_3, x="step", y="Accuracy", hue="exp", palette="viridis", ax=axes[2])
    sns.move_legend(axes[0], loc='lower right')
    sns.move_legend(axes[1], loc='lower right')
    sns.move_legend(axes[2], loc='lower right')
    plt.show()


def plot_test():
    fig, _ = plt.subplots()
    fig.suptitle("Test accuracy for each mode")
    x = ["0", "1", "2", "3"]
    y = [0.9070, 0.9103, 0.101199, 0.919900]
    clrs4 = ["palegreen", "skyblue", "salmon", "tomato"]
    sns.barplot(x=x, y=y, palette=clrs4)
    plt.show()

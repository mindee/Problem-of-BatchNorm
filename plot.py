import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("/home/laptopmindee/Documents/batchnorm.csv")

# Split and rename dataframes
df_True = df[["Step", "True-train", "True-val"]]
df_True = df_True.rename(columns={'True-train': 'Train', 'True-val': 'Val'})
df_False = df[["Step", "False-train", "False-val"]]
df_False = df_False.rename(columns={'False-train': 'Train', 'False-val': 'Val'})
df_Sans = df[["Step", "Sans-train", "Sans-val"]]
df_Sans = df_Sans.rename(columns={'Sans-train': 'Train', 'Sans-val': 'Val'})

dfm_True = df_True.melt('Step', var_name='exp', value_name='Accuracy')
dfm_False = df_False.melt('Step', var_name='exp', value_name='Accuracy')
dfm_Sans = df_Sans.melt('Step', var_name='exp', value_name='Accuracy')

sns.set_theme()

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15,5))
fig.suptitle("Train & Val accuracy on 10 epochs (MNIST) with and without BatchNorm")
axes[0].set_title("With running average")
axes[1].set_title("Without running average")
axes[2].set_title("Without BatchNorm")

sns.lineplot(data=dfm_True, x="Step", y="Accuracy", hue="exp", palette="flare", ax=axes[0])
sns.lineplot(data=dfm_False, x="Step", y="Accuracy", hue="exp", palette="flare", ax=axes[1])
sns.lineplot(data=dfm_Sans, x="Step", y="Accuracy", hue="exp", palette="flare", ax=axes[2])
plt.show()


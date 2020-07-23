import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns; sns.set(style='white')
import pandas as pd
import glob
import sys

# Check input
if "-pca" in sys.argv and "-heatmap" in sys.argv:
    print("Choose only pca or heatmap")
    sys.exit()

if "-pca" in sys.argv and "-batch_num" not in sys.argv and "-fault_ref" not in sys.argv:
    print("Choose '-batch-num' or '-fault_ref' as the target for PCA")
    sys.exit()

if "-pca" not in sys.argv and "-heatmap" not in sys.argv:
    print("Choose a valid option")
    sys.exit()

# Load all csvs
all_files = glob.glob("*.csv")
li = []
filenames = []

counter = 0
# Load all the datasets into one
for filename in all_files:
    filenames.append(filename)
    df = pd.read_csv(filename, index_col=None, header=0)
    temp = [counter] * len(df.index)
    # Tells us which file/batch the data is from
    df['Batch_Num'] = temp
    counter += 1
    li.append(df)

data = pd.concat(li, axis=0, ignore_index=True)
# Features being dropped
data = data.drop(columns=['RPM', 'X_offline', 'Viscosity_offline', 'P_offline', 'NH3_offline', 'PAA_offline', 'Batch ID', 'Batch #', 'Control_ref', 'Time (h)', 'NH3_shots', 'Batch_ref', 'PAT_ref'])

if "-heatmap" in sys.argv:
    ax = sns.heatmap(data.corr())
    ax.grid()
    plt.show()

features = data.columns.tolist()
# Separate dataset from target
x = data.loc[:, features].values
y = data.loc[:, ['Fault_ref']].values
y2 = data.loc[:, ['Batch_Num']].values
# Standardizing the data
x = StandardScaler().fit_transform(x)

if "-pca" in sys.argv:
    # PCA Stuff
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    print("PCA Variance: ", pca.explained_variance_ratio_)
    principalDf = pd.DataFrame(data=principalComponents, columns=['pc1', 'pc2'])

    finalDf = pd.concat([principalDf, data[['Fault_ref']]], axis=1)
    finalDf2 = pd.concat([principalDf, data[['Batch_Num']]], axis=1)

    # Pairplot the PCA result, not sure if this is needed
    #g = sns.pairplot(finalDf, hue="Fault_ref")


    # Plot results
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [1, 0]
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', '#FFE4E1', '#eeefff']

    if "-batch_num" in sys.argv:
        for i in range(len(li)):
            temp_df = finalDf2[finalDf2['Batch_Num'] == i]
            ax.scatter(temp_df['pc1'], temp_df['pc2'], c=colors[i], s=5)
            ax.legend(filenames)

    if "-fault_ref" in sys.argv:
        fault_df = finalDf[finalDf['Fault_ref'] == 1]
        good_df = finalDf[finalDf['Fault_ref'] == 0]
        print("Fault shape: ", fault_df.shape)
        print("Good shape: ", good_df.shape)
        ax.scatter(fault_df['pc1']
                   , fault_df['pc2']
                   , c='r'
                   , s=50)
        ax.scatter(good_df['pc1']
                   , good_df['pc2']
                   , c='g'
                   , s=50)

        ax.legend(targets)
    ax.grid()
    plt.show()
